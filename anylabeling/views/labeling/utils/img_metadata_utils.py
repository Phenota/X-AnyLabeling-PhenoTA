import json

from PIL import Image


def update_metadata_to_jpg_file(filename: str, metadata: dict) -> None:
    """
    update metadata to an JPG image file.
    """
    img = Image.open(filename)
    img.save(filename, comment=json.dumps(metadata))


def save_img_as_jpg_with_metadata(img: Image, filename: str, metadata: dict) -> None:
    """
    Save an image as a JPG file with metadata.
    """
    img.save(filename, comment=json.dumps(metadata))


def get_metadata_from_jpg_file(filename: str) -> dict:
    """
    get metadata from an JPG image file.
    """
    img = Image.open(filename)
    metadata = json.loads(img.info.get('comment', ''))
    return metadata


def get_metadata_from_img(img: Image) -> dict:
    """
    Get metadata from an image.
    """
    comment = img.info.get('comment')
    if comment:
        metadata = json.loads(comment)
    else:
        metadata = {}
    return metadata


if __name__ == '__main__':
    filename = './Cell1.jpg'
    img = Image.open(filename)
    print(get_metadata_from_img(img))
    metadata = {'name': 'Cell1', 'type': 'White Blood Cell'}
    save_img_as_jpg_with_metadata(img, filename, metadata)


