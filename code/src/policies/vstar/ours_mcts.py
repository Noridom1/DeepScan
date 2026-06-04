import base64, io, asyncio
import numpy as np
from PIL import Image
from typing import Tuple
from pathlib import Path
from .MCTS import MCTSQuestionSample, MCTSNode
from .client import get_heatmap
from .visual_grounding import grounding
from . import artifacts
from log import get_logger

logger = get_logger("ours")


class OursMCTSQuestionSample(MCTSQuestionSample):
    async def get_final_answer(self, use_mcts=True):
        if not hasattr(self, "_prefilter_done"):
                self._prefilter_done = True
                logger.info(
                    "grounding question_id=%s category=%r question=%r",
                    self.row.get('index'), self.category, self.row.get('question'),
                )
                if self.category == 'relative_position':
                    resized_img, resized_width, resized_height, objects_1 = grounding(self.image, self.row['question'], BLOCK=768, artifact_sink=self)
                else:
                    resized_img, resized_width, resized_height, objects_1 = grounding(self.image, self.row['question'], BLOCK=640, artifact_sink=self)
                logger.debug(
                    "grounding resized=%dx%d proposals=%d bboxes=%s",
                    resized_width, resized_height, len(objects_1),
                    [obj.get('bbox') for obj in objects_1],
                )

                flag, union_bbox = await self.justify(objects_1)
                self.flag = flag
                logger.info("justify flag=%s union_bbox=%s", flag, union_bbox)

                if artifacts.is_enabled(self):
                    stage_dir = artifacts.stage_dir(self, "hierarchical-scanning")
                    artifacts.save_json(stage_dir / "confirmed_union_resized.json", {
                        "flag": flag,
                        "union_bbox_resized": union_bbox,
                        "confirmed_count": len([obj for obj in objects_1 if obj.get('bbox')])
                    })

                if flag:
                    bbox_org  = self.convert_bbox_to_original_frame((0, 0, self.image_width, self.image_height), resized_width, resized_height, union_bbox)
                    logger.debug("justify original_frame_bbox=%s", bbox_org)
                    img_bytes = base64.b64decode(self.image)
                    groud_img = Image.open(io.BytesIO(img_bytes)).crop(bbox_org)
                    buffered = io.BytesIO()
                    groud_img.save(buffered, format="PNG")
                    groud_img_b64 = base64.b64encode(buffered.getvalue()).decode()

                    if artifacts.is_enabled(self):
                        stage_dir = artifacts.stage_dir(self, "hierarchical-scanning")
                        artifacts.save_json(stage_dir / "confirmed_union_original.json", {
                            "flag": flag,
                            "union_bbox_original": bbox_org,
                            "original_image_size": (self.image_width, self.image_height)
                        })
                        artifacts.save_base64_image(stage_dir / "07_confirmed_union_original.png", self.image)
                        artifacts.save_bbox_overlay(stage_dir / "07_confirmed_union_original_bbox.png", self.image, [bbox_org])
                        artifacts.save_base64_image(stage_dir / "08_initial_refocus_view.png", groud_img_b64)

                    # update initial state
                    self.initial_state = {
                        'depth': 0,
                        'image': groud_img_b64,
                        'action_history': [],
                        'text': self.row['question'],  # Root node uses original question as text
                        'image_width': self.image_width,
                        'image_height': self.image_height,
                        'region_coords': bbox_org
                    }

                else:
                    if artifacts.is_enabled(self):
                        stage_dir = artifacts.stage_dir(self, "hierarchical-scanning")
                        artifacts.save_json(stage_dir / "confirmed_union_original.json", {
                            "flag": flag,
                            "message": "No proposals accepted, using original image"
                        })
                        artifacts.save_base64_image(stage_dir / "08_initial_refocus_view.png", self.image)

                    self.initial_state = {
                        'depth': 0,
                        'image': self.image,
                        'action_history': [],
                        'text': self.row['question'],  # Root node uses original question as text
                        'image_width': self.image_width,
                        'image_height': self.image_height,
                        'region_coords': (0, 0, self.image_width, self.image_height)
                    }
                logger.debug(
                    "initial_state region=%s depth=%d",
                    self.initial_state['region_coords'], self.initial_state['depth'],
                )

        return await super().get_final_answer()


    async def justify(self, found_objs, TOP_K=None):
        def is_contained(box_a, box_b):
            ax1, ay1, ax2, ay2 = box_a
            bx1, by1, bx2, by2 = box_b
            return ax1 >= bx1 and ay1 >= by1 and ax2 <= bx2 and ay2 <= by2

        confirmed_bboxes = []
        model_name = Path(self.args.model_path).name
        if "qwen3-vl" in model_name.lower():
            prompt = f"""I will provide you an image and a question: {self.row['question']}. Please first review the provided images and determine whether the provided image contains the clues for answering the question or not. Give the brief reason and evidence of your decision, and then answer with **Yes** or **No."""
        else:
            prompt = f"""I will provide you an image and a **question** {self.row['question']}, please firstly determine wether the image contains the clues for answering the question or not (answer with **Yes** or **No**); then give the evidence of your decision."""

        # Save binary judgement prompt
        if artifacts.is_enabled(self):
            stage_dir = artifacts.stage_dir(self, "hierarchical-scanning")
            artifacts.save_text_file(stage_dir / "binary_judgement_prompt.txt", prompt)

        judgement_results = []
        if TOP_K:
            for idx, obj in enumerate(found_objs[: TOP_K]):
                image = obj['crop_img']
                bbox = obj['bbox']
                response = await self.generate_local(prompt, image, max_tokens=256)
                logger.debug(
                    "justify proposal=%d bbox=%s response=%r",
                    idx, bbox, response.replace('\n', ' ')[:300],
                )
                accepted = 'yes' in response.lower()
                if accepted:
                    confirmed_bboxes.append(bbox)

                if artifacts.is_enabled(self):
                    stage_dir = artifacts.stage_dir(self, "hierarchical-scanning/proposals")
                    artifacts.save_json(stage_dir / f"proposal_{idx:02d}_judgement.json", {
                        "proposal_index": idx,
                        "bbox_resized_frame": bbox,
                        "accepted": accepted,
                        "raw_response": response,
                        "prompt": prompt
                    })
        else:
            for idx, obj in enumerate(found_objs):
                image = obj['crop_img']
                bbox = obj['bbox']

                response = await self.generate_local(prompt, image, max_tokens=256)
                logger.debug(
                    "justify proposal=%d bbox=%s response=%r",
                    idx, bbox, response.replace('\n', ' ')[:300],
                )
                accepted = 'yes' in response.lower()
                if accepted:
                    confirmed_bboxes.append(bbox)

                if artifacts.is_enabled(self):
                    stage_dir = artifacts.stage_dir(self, "hierarchical-scanning/proposals")
                    artifacts.save_json(stage_dir / f"proposal_{idx:02d}_judgement.json", {
                        "proposal_index": idx,
                        "bbox_resized_frame": bbox,
                        "accepted": accepted,
                        "raw_response": response,
                        "prompt": prompt
                    })

        if not confirmed_bboxes:
            logger.debug("justify confirmed_bboxes=[]")
            return False, None

        if len(confirmed_bboxes) == 1:
            logger.debug("justify confirmed_bboxes=%s", confirmed_bboxes)
            return True, confirmed_bboxes[0]

        bboxes_array = np.array(confirmed_bboxes)
        x_min = np.min(bboxes_array[:, 0])
        y_min = np.min(bboxes_array[:, 1])
        x_max = np.max(bboxes_array[:, 2])
        y_max = np.max(bboxes_array[:, 3])

        union_bbox = (x_min, y_min, x_max, y_max)
        logger.debug("justify confirmed_bboxes=%s union=%s", confirmed_bboxes, union_bbox)
        return True, union_bbox


    def convert_bbox_to_original_frame(
        self,
        region_px: Tuple[int, int, int, int],
        resized_width: int,
        resized_height: int,
        union_bbox: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:

        x_p, y_p, x2_p, y2_p = region_px

        w_crop = x2_p - x_p
        h_crop = y2_p - y_p

        w_resized = resized_width
        h_resized = resized_height

        x_u, y_u, x2_u, y2_u = union_bbox

        if w_crop == 0 or h_crop == 0:
            raise ValueError("zero divide error")

        scale_x = w_resized / w_crop
        scale_y = h_resized / h_crop

        x_on_crop = x_u / scale_x
        y_on_crop = y_u / scale_y
        x2_on_crop = x2_u / scale_x
        y2_on_crop = y2_u / scale_y

        final_x1 = x_p + x_on_crop
        final_y1 = y_p + y_on_crop
        final_x2 = x_p + x2_on_crop
        final_y2 = y_p + y2_on_crop

        return tuple(map(int, (final_x1, final_y1, final_x2, final_y2)))
