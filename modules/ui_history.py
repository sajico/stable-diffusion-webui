import json
import os
from types import FunctionType
from typing import Dict, Iterable, List, TypeVar
import gradio as gr

import modules.paths as paths

from modules import shared

P = TypeVar("P")
R = TypeVar("R")
T = TypeVar("T")
history_file_name = "history.jsonl"
history_file_encodeing = "utf8"
params_file_name = "params.txt"
params_file_encodeing = "utf8"
mode_append = "a"
mode_read = "r"
mode_write = "w"
new_line = "\n"
comma = ","
colon = ":"
semi_colon = ";"
space = " "


class HistoryKeys:
    FullFn = "FullFn"
    Prompt = "Prompt"
    Negative_Prompt = "Negative prompt"
    Steps = "Steps"
    Sampler = "Sampler"
    CFG_Scale = "CFG scale"
    Image_CFG_scale = "Image CFG scale"
    Seed = "Seed"
    Face_restoration = "Face restoration"
    Size = "Size"
    Model_hash = "Model hash"
    Model = "Model"
    Variation_seed = "Variation seed"
    Variation_seed_strength = "Variation seed strength"
    Seed_resize_from = "Seed resize from"
    Conditional_mask_weight = "Conditional mask weight"
    Denoising_strength = "Denoising strength"
    Clip_skip = "Clip skip"
    ENSD = "ENSD"
    Token_merging_ratio = "Token merging ratio"
    Token_merging_ratio_hr = "Token merging ratio hr"
    Init_image_hash = "Init image hash"
    RNG = "RNG"
    NGMS = "NGMS"
    Hires_prompt = "Hires prompt"
    Hires_negative_prompt = "Hires negative prompt"
    Hires_upscale = "Hires upscale"
    Hires_upscaler = "Hires upscaler"
    Hires_sampler = "Hires sampler"
    Hires_resize = "Hires resize"
    Hires_steps = "Hires steps"
    Lora_hashes = "Lora hashes"
    Noise_multiplier = "Noise multiplier"
    Eta_DDIM = "Eta DDIM"
    Eta = "Eta"
    Discard_penultimate_sigma = "Discard penultimate sigma"
    Schedule_type = "Schedule type"
    Schedule_min_sigma = "Schedule min sigma"
    Schedule_max_sigma = "Schedule max sigma"
    Schedule_rho = "Schedule rho"
    Pad_conds = "Pad conds"
    Decode_prompt = "Decode prompt"
    Decode_negative_prompt = "Decode negative prompt"
    Decode_CFG_scale = "Decode CFG scale"
    Decode_steps = "Decode steps"
    Randomness = "Randomness"
    Sigma_Adjustment = "Sigma Adjustment"
    SD_upscale_overlap = "SD upscale overlap"
    SD_upscale_upscaler = "SD upscale upscaler"
    Script = "Script"
    X_Type = "X Type"
    X_Values = "X Values"
    Fixed_X_Values = "Fixed X Values"
    Y_Type = "Y Type"
    Y_Values = "Y Values"
    Fixed_Y_Values = "Fixed Y Values"
    Z_Type = "Z Type"
    Z_Values = "Z Values"
    Fixed_Z_Values = "Fixed Z Values"
    Version = "Version"


def ceil(a: int, b: int) -> int:
    return -1 * (-a // b)


def list_map(fn: FunctionType, itr: Iterable[P]) -> List[R]:
    return list(map(fn, itr))


def json_lines_to_dict_objects(jsonLines: List[str]) -> List[Dict]:
    return list_map(lambda jsonLine: json.loads(jsonLine), jsonLines)


def convert_dict_object_to_list_by_key_list(
    dict_object: Dict, key_list: List[str]
) -> List[List[any]]:
    return list_map(
        lambda key: dict_object[key] if key in dict_object else "", key_list
    )


def json_lines_to_data_frame_rows(
    json_lines: List[str], headers: List[str]
) -> List[List[any]]:
    return list_map(
        lambda data: convert_dict_object_to_list_by_key_list(data, headers),
        json_lines_to_dict_objects(json_lines),
    )


def key_value_class_to_value_list(T) -> List[str]:
    return list_map(
        lambda item: item[1],
        filter(lambda item: not "__" in item[0], T.__dict__.items()),
    )


def write_history(imagepath: str, infotext: str):
    with open(
        os.path.join(paths.data_path, history_file_name),
        mode_append,
        encoding=history_file_encodeing,
    ) as file:
        file.write(create_history_line(imagepath, infotext) + new_line)


def create_history_line(imagepath: str, infotext: str) -> str:
    infotext_split = infotext.splitlines()
    infoparams = map(
        lambda v: str(v).split(colon + space, 1),
        str(infotext_split[2]).split(comma + space),
    )
    history = {
        HistoryKeys.FullFn: imagepath,
        HistoryKeys.Prompt: infotext_split[0],
        HistoryKeys.Negative_Prompt: str(infotext_split[1]).split(colon + space)[1],
        **dict(infoparams),
    }
    return json.dumps(history)


class UiHistory:
    gallery_mode = False

    interface = None
    txt2img_button = None
    img2img_button = None
    first_button = None
    prev_button = None
    next_button = None
    last_button = None
    reload_button = None
    restart_button = None
    gallery = None

    history_keys = ["id", "image"] + key_value_class_to_value_list(HistoryKeys)
    history_extra_keys = history_keys[5:]
    history_value_type = list_map(lambda v: "str", history_keys)
    history_value_type[1] = "markdown"
    history_records = []
    selected_history = None

    gallery_per_page = 4
    gallery_page = 1
    gallery_last_page = 0
    gallery_min = 0
    gallery_max = gallery_per_page

    def get_gallery_values(self):
        gallery_values = self.history_records[self.gallery_min : self.gallery_max]
        if self.gallery_mode:
            gallery_values = list_map(lambda v: v[2], gallery_values)
        return gallery_values

    def set_gallery_page(self, page):
        self.gallery_page = page
        if (self.gallery_page < 1) or (self.gallery_last_page < self.gallery_page):
            self.gallery_page = 1
        self.gallery_min = self.gallery_per_page * (self.gallery_page - 1)
        self.gallery_max = self.gallery_per_page * self.gallery_page
        return (
            gr.update(value=self.get_gallery_values()),
            gr.update(interactive=self.gallery_page > 1),
            gr.update(interactive=self.gallery_page > 1),
            gr.update(interactive=self.gallery_page < self.gallery_last_page),
            gr.update(interactive=self.gallery_page < self.gallery_last_page),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    def first_gallery_page(self):
        return self.set_gallery_page(1)

    def prev_gallery_page(self):
        return self.set_gallery_page(self.gallery_page - 1)

    def next_gallery_page(self):
        return self.set_gallery_page(self.gallery_page + 1)

    def last_gallery_page(self):
        return self.set_gallery_page(self.gallery_last_page)

    def gallery_select(self, evt: gr.SelectData):
        history_index = evt.index if self.gallery_mode else evt.index[0]
        history_index += self.gallery_per_page * (self.gallery_page - 1)
        self.selected_history = self.history_records[history_index]
        return gr.update(interactive=True), gr.update(interactive=True)

    def read_history_jsonl(self):
        with open(
            os.path.join(paths.data_path, history_file_name),
            mode_read,
            encoding=history_file_encodeing,
        ) as file:
            historyRows = file.readlines()
            self.history_records = json_lines_to_data_frame_rows(
                historyRows, self.history_keys
            )
        self.gallery_last_page = ceil(len(self.history_records), self.gallery_per_page)
        for idx, history_record in enumerate(self.history_records):
            history_record[0] = idx + 1
            history_record[1] = f"""![](/file={history_record[2]})"""
        self.history_records = self.history_records[::-1]

    def reload_history(self):
        self.read_history_jsonl()
        return self.set_gallery_page(1)

    def get_selected_history_item(self, key, only_value=False):
        prefix = (key + colon + space) if not only_value else ""
        value = self.selected_history[self.history_keys.index(key)]
        return prefix + value if value != "" else ""

    def write_params(self):
        with open(
            os.path.join(paths.data_path, params_file_name),
            mode_write,
            encoding=params_file_encodeing,
        ) as file:
            params = (
                self.get_selected_history_item(HistoryKeys.Prompt, True)
                + new_line
                + self.get_selected_history_item(HistoryKeys.Negative_Prompt)
                + new_line
            )
            extra_params = filter(
                lambda v: v != "",
                [
                    self.get_selected_history_item(key)
                    for key in self.history_extra_keys
                ],
            )
            params += (comma + space).join(extra_params)
            file.write(params)

    def create_ui(self):
        self.read_history_jsonl()
        with gr.Blocks(analytics_enabled=False) as gradio_ui:
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        self.txt2img_button = gr.Button(
                            value="Paste to txt2img",
                            variant="primary",
                            elem_id="txt2img_paste",
                            interactive=False,
                        )
                        self.img2img_button = gr.Button(
                            value="Paste to img2img",
                            variant="primary",
                            elem_id="img2img_paste",
                            interactive=False,
                        )
                with gr.Column():
                    with gr.Row():
                        self.reload_button = gr.Button(
                            value="Reload Data",
                            variant="primary",
                            elem_id="reload_data",
                        )
                        self.restart_button = gr.Button(
                            value="Reload UI",
                            variant="primary",
                            elem_id="settings_restart_gradio",
                        )
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        self.first_button = gr.Button(
                            value="First page",
                            variant="primary",
                            elem_id="gallery_first_page",
                            interactive=False,
                        )
                        self.prev_button = gr.Button(
                            value="Prev page",
                            variant="primary",
                            elem_id="gallery_prev_page",
                            interactive=False,
                        )
                with gr.Column():
                    with gr.Row():
                        self.next_button = gr.Button(
                            value="Next page",
                            variant="primary",
                            elem_id="gallery_next_page",
                        )
                        self.last_button = gr.Button(
                            value="Last page ({})".format(self.gallery_last_page),
                            variant="primary",
                            elem_id="gallery_last_page",
                        )
            with gr.Row():
                with gr.Column():
                    if self.gallery_mode:
                        self.gallery = gr.Gallery(
                            value=self.get_gallery_values()
                        ).style(
                            columns=[6], rows=[2], object_fit="contain", height="auto"
                        )
                    else:
                        self.gallery = gr.DataFrame(
                            value=self.get_gallery_values(),
                            headers=self.history_keys,
                            datatype=self.history_value_type,
                            elem_id="history_dataframe",
                            elem_classes="history_dataframe",
                            wrap=True,
                        )

            buttons = [
                self.gallery,
                self.first_button,
                self.prev_button,
                self.next_button,
                self.last_button,
                self.txt2img_button,
                self.img2img_button,
            ]

            self.gallery.select(
                fn=self.gallery_select,
                inputs=None,
                outputs=[*buttons[-2:]],
            )

            self.first_button.click(
                fn=self.first_gallery_page,
                inputs=None,
                outputs=[*buttons],
                show_progress=False,
            )

            self.prev_button.click(
                fn=self.prev_gallery_page,
                inputs=None,
                outputs=[*buttons],
                show_progress=False,
            )

            self.next_button.click(
                fn=self.next_gallery_page,
                inputs=None,
                outputs=[*buttons],
                show_progress=False,
            )

            self.last_button.click(
                fn=self.last_gallery_page,
                inputs=None,
                outputs=[*buttons],
                show_progress=False,
            )

            self.reload_button.click(
                fn=self.reload_history,
                inputs=None,
                outputs=[*buttons],
                show_progress=False,
            )

            self.restart_button.click(
                fn=shared.state.request_restart,
                _js="restart_reload",
                inputs=[],
                outputs=[],
            )

            self.txt2img_button.click(
                fn=self.write_params,
                _js=None,
                inputs=None,
                outputs=None,
                show_progress=False,
            ).then(
                fn=None,
                _js="click_txt2img_paste",
                inputs=None,
                outputs=None,
                show_progress=False,
            )

            self.img2img_button.click(
                fn=self.write_params,
                _js=None,
                inputs=None,
                outputs=None,
                show_progress=False,
            ).then(
                fn=None,
                _js="click_img2img_paste",
                inputs=None,
                outputs=None,
                show_progress=False,
            )

        self.interface = gradio_ui
