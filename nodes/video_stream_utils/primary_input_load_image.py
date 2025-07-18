from comfy.nodes.package_typing import CustomNode


class PrimaryInputLoadImage(CustomNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image"
    CATEGORY = "image"
    
    def load_image(self, image):
        return (image,)


NODE_CLASS_MAPPINGS = {
    "PrimaryInputLoadImage": PrimaryInputLoadImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrimaryInputLoadImage": "Primary Input Load Image"
}
