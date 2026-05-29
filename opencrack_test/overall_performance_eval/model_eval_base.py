import os
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from overall_performance_eval.Metric import SegMetricEvaluator

def build_metric_path(json_path):
    """
    将:
      .../output/<model_name>/<file>.json
    转换为:
      .../output/<model_name>/metric/<file>.json
    """
    dir_path, file_name = os.path.split(json_path)
    metric_dir = os.path.join(dir_path, "metric")
    os.makedirs(metric_dir, exist_ok=True)
    return os.path.join(metric_dir, file_name)

def save_data_to_json(data_list, json_file_path):
   
    # 确保目录存在
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

    # 写入 JSON
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

    # print(f"样本数量：{len(data_list)}, 已保存到: {json_file_path}")

def get_json_data(json_path):
    # 读取 JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

class OSRS_Crack_Inference_Base(ABC):
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path
        self.model = self.load_model(model_path)
        self.evaluator = SegMetricEvaluator()

    @abstractmethod
    def load_model(self, model_path):
        pass

    @abstractmethod
    def one_image_inference(self, image_path, prompt, mask_save_path, _id):
        
        pass

    # 进行整个数据集的推理
    def full_image_inference(self, exp_name, dataset_root_path, osrs_crack_json_path, output_dir = "output"):

        save_path = os.path.join(output_dir, self.model_name, exp_name)
        os.makedirs(save_path, exist_ok=True)

        data = get_json_data(osrs_crack_json_path)
        inference_infos = []

        for index, item in enumerate(data):
            _id = item["id"]
            print(f"[{index+1}/{len(data)}] {_id}")
            image_path = item["image_path"]
            mask_path = item["mask_path"]
            prompts = item["prompts"]
            for i, prompt in enumerate(prompts):
                mask_save_path = os.path.join(save_path, f"{self.model_name}_{exp_name}_{_id}_prompt{i+1}.png")
                self.one_image_inference(os.path.join(dataset_root_path, image_path), prompt, mask_save_path, _id)
                inference_info = {
                    "id": f"{self.model_name}_{exp_name}_{_id}_prompt{i+1}.png",
                    "prompt": prompt,
                    "gt_mask": os.path.join(dataset_root_path, mask_path),
                    "pred_mask": mask_save_path
                }
                inference_infos.append(inference_info)
                print(mask_save_path)

       
        save_json_path = os.path.join(output_dir, self.model_name, f"{exp_name}_inference_infos.json")
        save_data_to_json(inference_infos, save_json_path)

    def full_crack_inference(self, exp_name, dataset_root_path, osrs_crack_json_path, output_dir="output"):
        save_path = os.path.join(output_dir, self.model_name, exp_name)
        os.makedirs(save_path, exist_ok=True)

        data = get_json_data(osrs_crack_json_path)
        inference_infos = []

        for index, item in enumerate(data):
            image_id = item["image_id"]
            print(f"[{index+1}/{len(data)}] {image_id}")
            image_path = item["image_path"]
            mask_path = item["mask_path"]
            
            mask_save_path = os.path.join(save_path, f"{self.model_name}_{exp_name}_{image_id}.png")
            self.one_image_inference(os.path.join(dataset_root_path, image_path), "crack", mask_save_path, image_id)
            
            inference_info = {
                "id": f"{self.model_name}_{exp_name}_{image_id}.png",
                "prompt": "crack",
                "gt_mask": os.path.join(dataset_root_path, mask_path),
                "pred_mask": mask_save_path
            }
            inference_infos.append(inference_info)
            print(mask_save_path)

        save_json_path = os.path.join(output_dir, self.model_name, f"{exp_name}_inference_infos.json")
        save_data_to_json(inference_infos, save_json_path)
 

    @staticmethod
    def all_type_eval_json_create(model_name, exp_name, dataset_root_path, osrs_crack_json_path, output_dir = "output"):
        
        overall = []
        geometry = []
        direction = []
        position = []
        extremal = []
        single_attribute = []
        two_attribute = []
        three_attribute = []
        single_instances = []
        two_instances = []
        multi_instances = []
        prompt_1 = []
        prompt_2 = []
        prompt_3 = []

        data = get_json_data(osrs_crack_json_path)

        for item in data:
            
            _id = item["id"]
            image_path = item["image_path"]
            num_instances = item["num_instances"]
            mask_path = item["mask_path"]
            prompts = item["prompts"]
            semantic_attribute = item["semantic_attribute"]
            semantic_type = item["semantic_type"]
            
            inference_infos = []
            for i, prompt in enumerate(prompts):
                inference_info = {
                    "id": f"{model_name}_{exp_name}_{_id}_prompt{i+1}.png",
                    "prompt": prompt,
                    "gt_mask": os.path.join(dataset_root_path, mask_path),
                    "pred_mask": os.path.join(output_dir, model_name, exp_name, f"{model_name}_{exp_name}_{_id}_prompt{i+1}.png")
                }
                inference_infos.append(inference_info)

            overall = overall + inference_infos

            if semantic_type == "geometry" and num_instances > 1:
                geometry = geometry + inference_infos
                single_attribute = single_attribute + inference_infos
            elif semantic_type == "direction" and num_instances > 1:
                direction = direction + inference_infos
                single_attribute = single_attribute + inference_infos
            elif semantic_type == "position" and num_instances > 1:
                position = position + inference_infos
                single_attribute = single_attribute + inference_infos
            elif semantic_type == "extremal" and num_instances > 1:
                extremal = extremal + inference_infos
                single_attribute = single_attribute + inference_infos
            elif semantic_type == "two_attribute":
                two_attribute = two_attribute + inference_infos
            elif semantic_type == "three_attribute":
                three_attribute = three_attribute + inference_infos
            else:
                pass

            if num_instances == 1:
                single_instances = single_instances + inference_infos
            elif num_instances == 2:
                two_instances = two_instances + inference_infos
            elif num_instances > 2:
                multi_instances = multi_instances + inference_infos
            else:
                assert False, f"{_id} 出现 num_instances 错误！"

            assert len(prompts) == 3, f"{_id} prompt 数量错误!"

            prompt_1.append(inference_infos[0])
            prompt_2.append(inference_infos[1])
            prompt_3.append(inference_infos[2])

        save_dict = {
            "overall": overall,
            "geometry": geometry, 
            "direction": direction, 
            "position": position, 
            "extremal": extremal, 
            "single_attribute": single_attribute, 
            "two_attribute": two_attribute, 
            "three_attribute": three_attribute, 
            "single_instances": single_instances, 
            "two_instances": two_instances, 
            "multi_instances": multi_instances, 
            "prompt_1": prompt_1, 
            "prompt_2": prompt_2, 
            "prompt_3": prompt_3, 
        }

        for k, v in save_dict.items():
            save_data_to_json(v, os.path.join(output_dir, model_name, f"{exp_name}_{k}.json"))

        # for k, v in save_dict.items():
        #     OSRS_Crack_Inference_Base.compute_metric(os.path.join(output_dir, model_name, f"{exp_name}_{k}.json"))
        OSRS_Crack_Inference_Base.compute_metric(os.path.join(output_dir, model_name, f"{exp_name}_overall.json"))
        # OSRS_Crack_Inference_Base.compute_metric(os.path.join(output_dir, model_name, f"{exp_name}_direction.json"))
        # OSRS_Crack_Inference_Base.compute_metric(os.path.join(output_dir, model_name, f"{exp_name}_position.json"))
        # OSRS_Crack_Inference_Base.compute_metric(os.path.join(output_dir, model_name, f"{exp_name}_extremal.json"))
        

    @staticmethod
    def compute_metric(json_path):
        SegMetricEvaluator(
            # max_size=544
        ).precompute_all_metrics_and_save(
            json_path=json_path,
            out_path=build_metric_path(json_path),
            tolerance_px=2,
            compute_auc=True,
            extra_meta={"model": "model", "split": "test"}
        )