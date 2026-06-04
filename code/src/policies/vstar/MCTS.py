from .policy import QuestionSample as BaseQuestionSample
from utils import is_none
from pathlib import Path
import shortuuid
import base64
import io
from PIL import Image
import numpy as np
import random
import math
import aiohttp
import traceback
import re
from . import artifacts
from log import get_logger

logger = get_logger("mcts")


def _short_trace_text(value, limit=300):
    text = "" if value is None else str(value)
    text = text.replace("\n", "\\n")
    if len(text) > limit:
        return text[:limit] + "..."
    return text


class MCTSNode:
    def __init__(self, state, parent=None, available_actions=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0
        self.leaf_reward = 0
        self.untried_actions = available_actions.copy() if available_actions else []
        self.expert_info = None
        self.valid_area_ratio = 1.0
        self.region_coords = state.get('region_coords', (0, 0, state['image_width'], state['image_height']))
        self.extra_info = {}


class MCTSQuestionSample(BaseQuestionSample):
    def __init__(self, row, args, round_idx=0):
        super().__init__(row, args, round_idx)

        image_bytes = base64.b64decode(self.image)
        img = Image.open(io.BytesIO(image_bytes))
        self.image_width, self.image_height = img.size
        blank_image = Image.new('RGB', (32, 32), color='white')
        buffered = io.BytesIO()
        blank_image.save(buffered, format="PNG")
        self.blank_image = base64.b64encode(buffered.getvalue()).decode()
        self.max_depth = 3
        self.c_puct = 1.0
        self.n_simulations = 6
        self.use_ensemble = True

        self.actions = [
            "repeat_question",
            "zoom_out"  # New zoom out action
        ]

        self.action_prompts = {
            "repeat_question": "Repeat the question.",
            "zoom_out": "Zoom out the region by 1.5x"  # New zoom out action prompt
        }

        self.action_executors = {
            "repeat_question": self.execute_repeat_question_action,
            "zoom_out": self.execute_zoom_out_action  # New zoom out action executor
        }

        self.root = None
        self.expert_ports = [2] # <----- Changed the port
        self.expert_ports = [port + 8000 for port in self.expert_ports]
        self.expert_base_url = "http://localhost:{}/predict"
        self.artifact_step_idx = 0
        logger.info(
            "init question_id=%s image_size=%dx%d expert_ports=%s",
            self.row.get('index'), self.image_width, self.image_height, self.expert_ports,
        )


    async def extract_key_objects(self):
        if 'llava' in self.args.model_path:
            question = self.row['question'].replace('?', '')
            stop_words = ['is', 'in the image', 'IS', 'THE', 'IMAGE','what','color of','there', 'a', 'an', 'How']
            for word in stop_word:
                question = re.sub(r'\b' + word + r'\b', '', question, flags=re.IGNORECASE)
            question = ' '.join(question.split())

            prompt = f"Task: List objects mentioned in text in List format.\nInput text: {question}\nAction: What objects are mentioned in original text? List separated by commas. For example, from \"person with white trousers on the left or right side of the person in blue\", output \"[\"person with white trousers\", \"person in blue\"]\"."
            response = await self.generate_local(prompt, self.blank_image, max_tokens=50)
            logger.debug("key_objects raw_response=%r", _short_trace_text(response))

            try:
                objects = eval(response)
            except:
                response = response.replace('[', '').replace(']', '').replace('"', '')
                objects = response.split(',')

            filtered_objects = []
            for obj in objects:
                obj = obj.strip().lower()
                if obj in question.lower():
                    filtered_objects.append(obj)

            if not filtered_objects:
                filtered_objects = [question]

            return filtered_objects

        else:
            prompt = f"Task: Extract all objects (including people) with their complete descriptions from the question. For example, from 'Is the person with white trousers on the left or right side of the person in blue?', extract 'person with white trousers' and 'person in blue'.\nQuestion: {self.row['question']}\nAction: Only list the objects separated by commas."
            response = await self.generate_local(prompt, self.blank_image, max_tokens=50)
            logger.debug("key_objects raw_response=%r", _short_trace_text(response))

            if "object" in response.lower() or "description" in response.lower():
                objects = response.split()[-1].lower()

            objects = [obj.strip() for obj in response.split(',')]
            logger.info("key_objects parsed=%s", objects)
            return objects


    async def get_expert_boxes(self, image, text):
        try:
            port = random.choice(self.expert_ports)

            expert_url = self.expert_base_url.format(port)
            timeout = aiohttp.ClientTimeout(total=10000)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    expert_url,
                    json={
                        "image": image,  # image is already base64 string
                        "text": text
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.debug(
                            "visual_expert url=%s text=%r boxes=%d",
                            expert_url, _short_trace_text(text, 120), len(result.get('boxes', [])),
                        )
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(
                            "visual_expert error status=%s url=%s text=%r message=%s",
                            response.status, expert_url, text, error_text,
                        )
                        return None
        except Exception as e:
            logger.error(
                "visual_expert exception url=%s text=%r error=%s\n%s",
                expert_url, text, e, traceback.format_exc(),
            )
            return None


    def selection(self, node):
        if node.untried_actions:
            return node

        if not node.children:
            return node

        total_visits = sum(child.visits for child in node.children.values())

        def ucb_score(child):
            exploit = child.value / child.visits if child.visits > 0 else 0
            explore = math.sqrt(2 * math.log(total_visits) / (child.visits + 1e-8))
            return exploit + self.c_puct * explore

        best_child = max(node.children.values(), key=ucb_score)
        return self.selection(best_child)


    async def execute_repeat_question_action(self, node):
        step_idx = self.artifact_step_idx
        self.artifact_step_idx += 1

        node_text = self.row['question']
        expert_result = await self.get_expert_boxes(node.state['image'], node_text)

        if expert_result and expert_result.get('boxes'):
            boxes = np.array(expert_result['boxes'])

            x1 = np.min(boxes[:, 0])
            y1 = np.min(boxes[:, 1])
            x2 = np.max(boxes[:, 2])
            y2 = np.max(boxes[:, 3])

            padding = 32
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(node.state['image_width'], x2 + padding)
            y2 = min(node.state['image_height'], y2 + padding)

            new_area = (x2 - x1) * (y2 - y1)
            total_area = node.state['image_width'] * node.state['image_height']
            valid_area_ratio = new_area / total_area

            parent_x1, parent_y1, _, _ = node.state['region_coords']
            new_region_coords = (
                parent_x1 + x1,
                parent_y1 + y1,
                parent_x1 + x2,
                parent_y1 + y2
            )

            image_bytes = base64.b64decode(self.image)
            img = Image.open(io.BytesIO(image_bytes))
            cropped_img = img.crop(new_region_coords)

            buffered = io.BytesIO()
            cropped_img.save(buffered, format="PNG")
            cropped_image_base64 = base64.b64encode(buffered.getvalue()).decode()
            logger.debug(
                "action repeat_question boxes=%d region=%s valid_area_ratio=%.4f",
                len(expert_result.get('boxes', [])), new_region_coords, valid_area_ratio,
            )

        else:
            cropped_image_base64 = node.state['image']
            valid_area_ratio = node.valid_area_ratio
            new_region_coords = node.state['region_coords']
            logger.debug(
                "action repeat_question no_boxes region=%s valid_area_ratio=%.4f",
                new_region_coords, valid_area_ratio,
            )

        new_state = {
            'depth': node.state['depth'] + 1,
            'image': cropped_image_base64,
            'action_history': node.state['action_history'] + [self.action_prompts["repeat_question"]],
            'text': node_text,
            'image_width': node.state['image_width'],
            'image_height': node.state['image_height'],
            'region_coords': new_region_coords
        }

        child = MCTSNode(new_state, parent=node, available_actions=self.actions)
        child.expert_info = expert_result
        child.valid_area_ratio = valid_area_ratio

        if artifacts.is_enabled(self):
            sdir = artifacts.stage_dir(self, f"refocusing/step_{step_idx:03d}_repeat_question")
            artifacts.save_base64_image(sdir / "parent.png", node.state['image'])
            if expert_result and expert_result.get('boxes'):
                artifacts.save_bbox_overlay(
                    sdir / "expert_overlay.png",
                    node.state['image'],
                    [tuple(map(int, b)) for b in expert_result['boxes']],
                )
            artifacts.save_base64_image(sdir / "child.png", cropped_image_base64)
            artifacts.save_json(sdir / "step.json", {
                "step": step_idx,
                "action": "repeat_question",
                "parent_depth": node.state['depth'],
                "parent_region_coords": node.state['region_coords'],
                "expert_boxes_parent_frame": expert_result.get('boxes') if expert_result else None,
                "expert_boxes_found": bool(expert_result and expert_result.get('boxes')),
                "child_region_coords": new_region_coords,
                "valid_area_ratio": valid_area_ratio,
                "action_history": new_state['action_history'],
            })

        return child


    async def execute_zoom_out_action(self, node):
        step_idx = self.artifact_step_idx
        self.artifact_step_idx += 1

        x1, y1, x2, y2 = node.state['region_coords']

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        width = x2 - x1
        height = y2 - y1

        new_width = width * 1.5
        new_height = height * 1.5

        new_x1 = max(0, center_x - new_width/2)
        new_y1 = max(0, center_y - new_height/2)
        new_x2 = min(node.state['image_width'], center_x + new_width/2)
        new_y2 = min(node.state['image_height'], center_y + new_height/2)

        image_bytes = base64.b64decode(self.image)
        img = Image.open(io.BytesIO(image_bytes))
        cropped_img = img.crop((new_x1, new_y1, new_x2, new_y2))

        buffered = io.BytesIO()
        cropped_img.save(buffered, format="PNG")
        cropped_image_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Capture the 1.5x candidate before it may be overwritten by expert correction.
        zoom_candidate_b64 = cropped_image_base64
        zoom_candidate_region = (new_x1, new_y1, new_x2, new_y2)

        final_x1, final_y1, final_x2, final_y2 = new_x1, new_y1, new_x2, new_y2

        if 'missing_objects' in node.state and node.state['missing_objects']:
            expert_mode = "missing_objects"
            expert_query = ', '.join(node.state['missing_objects'])
            expert_result = await self.get_expert_boxes(cropped_image_base64, expert_query)

            if expert_result and expert_result.get('boxes'):
                boxes = np.array(expert_result['boxes'])

                expert_x1 = np.min(boxes[:, 0]) + new_x1
                expert_y1 = np.min(boxes[:, 1]) + new_y1
                expert_x2 = np.max(boxes[:, 2]) + new_x1
                expert_y2 = np.max(boxes[:, 3]) + new_y1

                final_x1 = min(x1, expert_x1)
                final_y1 = min(y1, expert_y1)
                final_x2 = max(x2, expert_x2)
                final_y2 = max(y2, expert_y2)

                cropped_img = img.crop((final_x1, final_y1, final_x2, final_y2))
                buffered = io.BytesIO()
                cropped_img.save(buffered, format="PNG")
                cropped_image_base64 = base64.b64encode(buffered.getvalue()).decode()
        else:
            expert_mode = "key_objects"
            expert_query = ", ".join(self.key_objects)
            expert_result = await self.get_expert_boxes(cropped_image_base64, expert_query)

        new_state = {
            'depth': node.state['depth'] + 1,
            'image': cropped_image_base64,
            'action_history': node.state['action_history'] + [self.action_prompts["zoom_out"]],
            'text': node.state['text'],
            'image_width': node.state['image_width'],
            'image_height': node.state['image_height'],
            'region_coords': (final_x1, final_y1, final_x2, final_y2)
        }

        child = MCTSNode(new_state, parent=node, available_actions=self.actions)
        child.expert_info = expert_result

        new_area = (final_x2 - final_x1) * (final_y2 - final_y1)
        total_area = node.state['image_width'] * node.state['image_height']
        child.valid_area_ratio = new_area / total_area
        logger.debug(
            "action zoom_out region=%s valid_area_ratio=%.4f boxes=%d",
            (final_x1, final_y1, final_x2, final_y2),
            child.valid_area_ratio,
            len(expert_result.get('boxes', [])) if expert_result else 0,
        )

        if artifacts.is_enabled(self):
            sdir = artifacts.stage_dir(self, f"refocusing/step_{step_idx:03d}_zoom_out")
            artifacts.save_base64_image(sdir / "parent.png", node.state['image'])
            artifacts.save_base64_image(sdir / "zoom_candidate.png", zoom_candidate_b64)
            if expert_result and expert_result.get('boxes'):
                artifacts.save_bbox_overlay(
                    sdir / "expert_overlay.png",
                    zoom_candidate_b64,
                    [tuple(map(int, b)) for b in expert_result['boxes']],
                )
            artifacts.save_base64_image(sdir / "child.png", cropped_image_base64)
            artifacts.save_json(sdir / "step.json", {
                "step": step_idx,
                "action": "zoom_out",
                "parent_depth": node.state['depth'],
                "parent_region_coords": node.state['region_coords'],
                "zoom_candidate_region_coords": zoom_candidate_region,
                "expert_mode": expert_mode,
                "expert_query": expert_query,
                "expert_boxes_zoom_candidate_frame": expert_result.get('boxes') if expert_result else None,
                "expert_boxes_found": bool(expert_result and expert_result.get('boxes')),
                "child_region_coords": (final_x1, final_y1, final_x2, final_y2),
                "valid_area_ratio": child.valid_area_ratio,
                "action_history": new_state['action_history'],
            })

        return child


    async def expansion(self, node):
        if node.state['depth'] >= self.max_depth or not node.untried_actions:
            return node

        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)
        child = await self.action_executors[action](node)
        node.children[action] = child

        return child


    async def simulation(self, node):
        key_objects = self.key_objects
        all_objects_present = True
        confirmed_objects = []
        missing_objects = []
        for obj in key_objects:
            prompt = f"Task: Only answer yes or no.\nQuestion: Is there a {obj} in this image?"
            response = await self.generate_local(prompt, node.state['image'], max_tokens=10)
            logger.debug(
                "simulation depth=%d obj=%r response=%r",
                node.state['depth'], obj, _short_trace_text(response, 80),
            )

            if 'yes' in response.lower():
                confirmed_objects.append(obj)
            else:
                missing_objects.append(obj)
                all_objects_present = False
                break

        node.state['caption'] = ', '.join(confirmed_objects)
        node.state['missing_objects'] = missing_objects
        if all_objects_present:
            reward = 1 - node.valid_area_ratio
        else:
            reward = 0
        logger.debug(
            "simulation depth=%d confirmed=%s missing=%s reward=%.4f",
            node.state['depth'], confirmed_objects, missing_objects, reward,
        )

        return reward


    def backpropagation(self, node, reward):
        """Backpropagation phase: update node values"""
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent


    async def single_run(self, root_state):
        if not self.root:
            temp_root = MCTSNode(root_state, available_actions=self.actions)
            self.root = await self.execute_repeat_question_action(temp_root)
            self.root.parent = None

        node = self.selection(self.root)

        if node.state['depth'] >= self.max_depth:
            return 0

        node = await self.expansion(node)
        reward = await self.simulation(node)
        node.leaf_reward = reward
        self.backpropagation(node, reward)

        return reward


    async def get_final_answer(self):
        """Run MCTS to search for best answer"""
        # Save initial state for refocusing
        if artifacts.is_enabled(self):
            stage_dir = artifacts.stage_dir(self, "refocusing")
            artifacts.save_base64_image(stage_dir / "00_initial_view.png", self.initial_state['image'])
            artifacts.save_json(stage_dir / "00_initial_view.json", {
                "depth": self.initial_state['depth'],
                "region_coords": self.initial_state['region_coords'],
                "image_width": self.initial_state['image_width'],
                "image_height": self.initial_state['image_height']
            })

        for sim_idx in range(self.n_simulations):
            logger.debug("search simulation %d/%d", sim_idx + 1, self.n_simulations)
            await self.single_run(self.initial_state)

        all_nodes = []
        nodes_to_visit = [self.root]
        while nodes_to_visit:
            node = nodes_to_visit.pop()
            all_nodes.append(node)
            nodes_to_visit.extend(node.children.values())

        if artifacts.is_enabled(self):
            stage_dir = artifacts.stage_dir(self, "refocusing")
            artifacts.save_search_tree(stage_dir / "search_tree.json", all_nodes)

        final_qs = ''
        if not is_none(self.row['hint']):
            final_qs += self.row['hint'] + '\n'
        final_qs += self.row['question']

        for option_char, option in zip(self.cur_option_char, self.options):
            final_qs += '\n' + option_char + '. ' + option

        if self.args.single_pred_prompt:
            if self.args.lang == 'cn':
                final_qs += '\n' + "仔细查看输入的图像以及放大后的证据（可选），然后直接从给出的选项中选择对应的字母来回答问题。"
            else:
                final_qs += '\n' + "Carefully review the input images as well as the zoomed-in evidence (optional), and then answer the question with the option's letter from the given choices directly."

        answers = []
        model_name = Path(self.args.model_path).name
        logger.info(
            "answer nodes=%d model_name=%r flag=%s",
            len(all_nodes), model_name, getattr(self, 'flag', None),
        )
        for node_idx, node in enumerate(all_nodes):
            if "qwen3-vl" in model_name.lower():
                answer = await self.generate_local(final_qs, node.state['image'])
                reasoning_images = [node.state['image']]
            else:
                 # multi-scale evidence enhanced reasoning
                if self.flag:
                    if node.state['image'] == self.initial_state['image']:
                        answer = await self.generate_local(final_qs, [node.state['image'], self.image])
                        reasoning_images = [node.state['image'], self.image]
                    else:
                        answer = await self.generate_local(final_qs, [node.state['image'], self.initial_state['image'], self.image])
                        reasoning_images = [node.state['image'], self.initial_state['image'], self.image]
                else:
                    if node.state['image'] == self.image:
                        answer = await self.generate_local(final_qs, node.state['image'])
                        reasoning_images = [node.state['image']]
                    else:
                        answer = await self.generate_local(final_qs, [node.state['image'], self.image])
                        reasoning_images = [node.state['image'], self.image]

            logger.debug(
                "answer node=%d depth=%d reward=%.4f region=%s raw=%r",
                node_idx, node.state['depth'], node.leaf_reward,
                node.state.get('region_coords'), _short_trace_text(answer),
            )

            # Export reasoning artifacts
            if artifacts.is_enabled(self):
                stage_dir = artifacts.stage_dir(self, f"reasoning/node_{node_idx:02d}")
                for img_idx, img_b64 in enumerate(reasoning_images):
                    artifacts.save_base64_image(stage_dir / f"input_{img_idx:02d}.png", img_b64)
                artifacts.save_text_file(stage_dir / "prompt.txt", final_qs)
                artifacts.save_text_file(stage_dir / "response.txt", answer)
                artifacts.save_json(stage_dir / "metadata.json", {
                    "node_index": node_idx,
                    "depth": node.state['depth'],
                    "region_coords": node.state.get('region_coords'),
                    "leaf_reward": node.leaf_reward,
                    "valid_area_ratio": node.valid_area_ratio,
                    "action_history": node.state.get('action_history', []),
                    "model_name": model_name,
                    "num_reasoning_images": len(reasoning_images),
                    "raw_response": answer
                })

            for letter in ['A', 'B', 'C', 'D']:
                if letter in answer:
                    answers.append((letter, node.leaf_reward))  # Use leaf reward as weight
                    break
            else:
                answers.append(('A', node.leaf_reward))  # If no valid option found, default to A with leaf reward

        best_node = max(all_nodes, key=lambda x: (x.leaf_reward, all_nodes.index(x)))

        if self.use_ensemble:
            from collections import defaultdict
            vote_result = defaultdict(float)
            for answer, weight in answers:
                vote_result[answer] += weight
            logger.info("vote weighted_votes=%s answers=%s", dict(vote_result), answers)

            if artifacts.is_enabled(self):
                stage_dir = artifacts.stage_dir(self, "reasoning")
                artifacts.save_json(stage_dir / "weighted_votes.json", {
                    "votes": dict(vote_result),
                    "all_answers": answers
                })

            if all(weight == 0 for weight in vote_result.values()):
                answer = await self.generate_local(final_qs, self.image)
                logger.debug("vote fallback_raw=%r", _short_trace_text(answer))
                for letter in ['A', 'B', 'C', 'D']:
                    if letter in answer:
                        final_answer = letter
                        break
                else:
                    final_answer = 'A'
            else:
                final_answer = max(vote_result, key=vote_result.get)
        else:
            final_answer = max(answers, key=lambda x: x[1])[0]

        logger.info(
            "final final_answer=%r best_reward=%.4f best_region=%s",
            final_answer, best_node.leaf_reward, best_node.state.get('region_coords'),
        )

        # Export final answer artifacts
        if artifacts.is_enabled(self):
            stage_dir = artifacts.stage_dir(self, "reasoning")
            artifacts.save_base64_image(stage_dir / "best_node.png", best_node.state['image'])
            artifacts.save_text_file(stage_dir / "final_prompt.txt", final_qs)
            artifacts.save_json(stage_dir / "final_answer.json", {
                "final_answer": final_answer,
                "best_node_index": all_nodes.index(best_node),
                "best_node_reward": best_node.leaf_reward,
                "best_node_region": best_node.state.get('region_coords'),
                "total_nodes_explored": len(all_nodes),
                "use_ensemble": self.use_ensemble
            })

        return final_answer, final_qs, answers[-1][0], best_node.state['image'], best_node, self.root


    def serialize_tree(self, node):
        """Serialize tree structure for saving to jsonl"""
        node_info = {
            "state": node.state,
            "visits": node.visits,
            "value": node.value,
            "leaf_reward": node.leaf_reward,
            "expert_info": node.expert_info,
            "valid_area_ratio": node.valid_area_ratio,
            "region_coords": node.region_coords,
            "extra_info": node.extra_info,
            "children": {action: self.serialize_tree(child) for action, child in node.children.items()}
        }
        return node_info


    async def _process(self):
        self.key_objects = await self.extract_key_objects()
        logger.info("process key_objects=%s", self.key_objects)

        final_answer, prompt, full_answer, final_image, best_node, root_node = await self.get_final_answer()
        logger.info(
            "process final_answer=%r full_answer=%r gold=%r",
            final_answer, full_answer, self.row['answer'],
        )

        # Serialize tree structure for saving
        # tree_info = self.serialize_tree(best_node)

        return {
            "question_id": self.row['index'],
            "round_id": self.round_idx,
            "prompt": prompt,
            "text": final_answer,
            "options": self.options,
            "option_char": self.cur_option_char,
            "answer_id": shortuuid.uuid(),
            "model_id": self.args.model_path,
            "answer": self.row['answer'],
        }
