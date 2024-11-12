"""Shared class mapping utilities"""

def get_class_from_filename(filename):
    """Extract class number from filename and return corresponding class name"""
    class_num = filename.split('_')[0]
    return CLASS_MAPPING.get(class_num, 'unknown')

CLASS_MAPPING = {
    '1': 'farm animals',
    '2': 'trees',
    '3': 'buildings',
    '4': 'airplanes',
    '5': 'cows',
    '6': 'faces',
    '7': 'cars',
    '8': 'bicycles',
    '9': 'sheep',
    '10': 'flowers',
    '11': 'signs',
    '12': 'birds',
    '13': 'books',
    '14': 'chairs',
    '15': 'cats',
    '16': 'dogs',
    '17': 'street',
    '18': 'nature',
    '19': 'people',
    '20': 'boats',
} 