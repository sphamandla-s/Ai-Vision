from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from dotenv import load_dotenv
import os
import sys


# Get Configuration Settings
load_dotenv()
ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
ai_key = os.getenv('AI_SERVICE_KEY')

# Authenticate Azure AI Vision client
cv_client = ImageAnalysisClient(ai_endpoint, AzureKeyCredential(ai_key))


def main():
    try:
        # Get image
        image_file = 'images/street.jpg'
        if len(sys.argv) > 1:
            image_file = sys.argv[1]

        # Check if the file exists before opening
        if not os.path.exists(image_file):
            print(f"Error: File '{image_file}' not found.")
            return

        with open(image_file, "rb") as f:
            image_data = f.read()
        AnalyzeImage(image_file, image_data, cv_client)

    except Exception as ex:
        print(f"An error occurred: {ex}")


def AnalyzeImage(image_filename, image_data, cv_client):
    print(f"Analyzing image: {image_filename}")

    try:

        features = [
            VisualFeatures.TAGS,
            VisualFeatures.OBJECTS,
            VisualFeatures.PEOPLE
        ]

        # Get result with specified features to be retrieved
        result = cv_client.analyze(
            image_data=image_data,
            visual_features=features
        )

    except HttpResponseError as e:
        print(f"Status code: {e.status_code}")
        print(f"Reason: {e.reason}")
        print(f"Message: {e.error.message}")

    if result.objects is not None:
        print('\nObjects in image:')

        # Prepare image for drawing
        image = Image.open(image_filename)
        fig = plt.figure(figsize=(image.width/100, image.height/100))
        plt.axis('off')
        draw = ImageDraw.Draw(image)
        color = 'red'

        for detected_object in result.objects.list:
            # Print object name
            print(" {} (confidence: {:.2f}%)".format(
                detected_object.tags[0].name, detected_object.tags[0].confidence * 100))

           # Draw object bounding box
            r = detected_object.bounding_box
            bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
            draw.rectangle(bounding_box, outline=color, width=3)
            plt.annotate(
                detected_object.tags[0].name, (r.x, r.y), backgroundcolor=color)

        # Save annotate image
        plt.imshow(image)
        plt.tight_layout(pad=0)
        outputfile = 'objects.jpg'
        fig.savefig(outputfile)
        print('  Results saved in', outputfile)
        image.show(outputfile)


if __name__ == "__main__":
    main()
