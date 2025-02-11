from .nodes import TextNode, CosyVoiceNode, LoadSRT, CosyVoiceDubbingNode, CosyVoiceSRT2Speech

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "LoadSRT": LoadSRT,
    "TextNode": TextNode,
    "CosyVoiceNode": CosyVoiceNode,
    "CosyVoiceDubbingNode": CosyVoiceDubbingNode,
    "CosyVoiceSRT2Speech": CosyVoiceSRT2Speech
}
