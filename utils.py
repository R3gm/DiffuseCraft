import gradio as gr
from dartrs.v2 import AspectRatioTag, LengthTag, RatingTag, IdentityTag


V2_ASPECT_RATIO_OPTIONS: list[AspectRatioTag] = [
    "ultra_wide",
    "wide",
    "square",
    "tall",
    "ultra_tall",
]
V2_RATING_OPTIONS: list[RatingTag] = [
    "sfw",
    "general",
    "sensitive",
    "nsfw",
    "questionable",
    "explicit",
]
V2_LENGTH_OPTIONS: list[LengthTag] = [
    "very_short",
    "short",
    "medium",
    "long",
    "very_long",
]
V2_IDENTITY_OPTIONS: list[IdentityTag] = [
    "none",
    "lax",
    "strict",
]


# ref: https://qiita.com/tregu148/items/fccccbbc47d966dd2fc2
def gradio_copy_text(_text: None):
    gr.Info("Copied!")


COPY_ACTION_JS = """\
(inputs, _outputs) => {
  // inputs is the string value of the input_text
  if (inputs.trim() !== "") {
    navigator.clipboard.writeText(inputs);
  }
}"""


def gradio_copy_prompt(prompt: str):
    gr.Info("Copied!")
    return prompt
