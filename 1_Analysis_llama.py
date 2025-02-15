import os
import base64
import logging
from PIL import Image
import io
from typing import Dict, Optional
from groq import Groq

class SkinAnalyzer:
    """
    A class to analyze skin conditions using Llama 3.3-Vision API via Groq with streaming responses.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the SkinAnalyzer with Groq API credentials.
        """
        os.environ["GROQ_API_KEY"] = api_key
        self.client = Groq()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def encode_image(self, image_path: str) -> Optional[str]:
        """
        Encode the image file to base64 string after resizing and compressing.
        """
        try:
            with Image.open(image_path) as img:
                img = img.resize((128, 128), Image.Resampling.LANCZOS)  # Resizing the image
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=60)  # Compressing the image
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Error encoding image: {str(e)}")
            return None

    def analyze_skin_condition(self, image_path: str) -> str:
        """
        Analyze skin condition in the provided image using Groq's Llama 3.3 API with streaming.
        """
        image_data = self.encode_image(image_path)
        if not image_data:
            return "Error: Failed to encode image"

        messages = [
            {
                "role": "system",
                "content": ("Give a 2-3 word final diagnosis of which condition from the list mentioned by the user you believe is most likely present in the image."
                            " Act as a professional dermatologist doctor. What's in this image? Do you find anything wrong with the face in terms of a dermatologist?"
                            " If you make a differential, suggest some remedies for them."
                            " Your response should be in two different paragraphs, one where you tell what is wrong with the face and the other where you make a skincare plan for the user based on his skin diagnosis."
                            " Always answer as if you are addressing a real person. No preamble, start your answer immediately.")
            },
            {
                "role": "user",
                "content": f"data:image/jpeg;base64,{image_data}"
                " Please examine the image for the presence of any of the following skin conditions:"
                           "\n1. Acne: Look for red, pus-filled pimples typical of acne as seen on highly textured, inflamed skin surfaces."
                           "\n2. Actinic Keratosis: Identify scaly, crusty patches that show rough texture on sun-exposed areas."
                           "\n3. Basal Cell Carcinoma: Check for waxy or shiny bumps, often with visible blood vessels or a pearly appearance, typically located on facial regions."
                           "\n4. Eczema: Look for areas with inflamed, itchy, and red skin, often dry or flaky, indicative of eczema."
                           "\n5. Herpes: Identify blister-like sores, clustered and typically filled with liquid, common in herpes infections."
                           "\n6. Panu (Tinea Versicolor): Search for patches of discolored skin that may appear tan, pink, or white, often irregular and scattered."
                           "\n7. Rosacea: Look for persistent redness, visible blood vessels, or small red bumps on the face, characteristic of rosacea."
                           "\n\nProvide a clear identification of any conditions found, including their distinct visual characteristics and the specific areas of the skin they affect." 
                           "If multiple conditions are visible, please describe each separately and note any overlapping features or interactions between conditions."
            }
        ]

        try:
            completion = self.client.chat.completions.create(
                model="gemma2-9b-it",
                messages=messages,
                temperature=0.6,
                max_completion_tokens=500,
                top_p=1,
                stream=True,
                stop=None
            )

            # Collecting and formatting the stream of responses
            analysis_result = ""
            for chunk in completion:
                analysis_result += chunk.choices[0].delta.content or ""
            
            return analysis_result

        except Exception as e:
            self.logger.error(f"Error during analysis: {str(e)}")
            return f"Error: {str(e)}"

def main():
    """
    Example usage of the SkinAnalyzer class.
    """
    api_key = "USE-YOUR-OWN-GODDAMN-KEY"
    analyzer = SkinAnalyzer(api_key)
    image_path = r"basal-cell-carcinoma-face-20.jpg"

    analysis_results = analyzer.analyze_skin_condition(image_path)
    
    if "Error" in analysis_results:
        print(analysis_results)
    else:
        print("\nAnalysis Results:")
        print("------------------------------")
        print(analysis_results.replace("\n", "\n\n"))
        print("------------------------------")

if __name__ == "__main__":
    main()