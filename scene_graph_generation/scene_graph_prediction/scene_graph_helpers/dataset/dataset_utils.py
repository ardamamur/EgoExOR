import re

SOURCES = {
    "head_surgeon": 1,
    "assistant": 2,
    "circulator": 3,
    "anesthetist": 4,
    "or_light": 5,
    "microscope": 6,
    "external_1": 7,
    "external_2": 8,
    "external_3": 9,
    "external_4": 10,
    "external_5": 11,
    "simstation": 12,
    "ultrasound": 13,
    "blank": -1
}

ENTITY_VOCAB = {
    'anaesthetist': 0,
    'anesthesia_equipment': 1,
    'antiseptic': 2,
    'assistant': 3,
    'bin': 4,
    'body_marker': 5,
    'circulator': 6,
    'cotton': 7,
    'curette': 8,
    'dressing_material': 9,
    'forceps': 10,
    'gloves': 11,
    'head_surgeon': 12,
    'health_monitor': 13,
    'herbal_disk': 14,
    'instrument_table': 15,
    'instruments': 16,
    'microcope': 17, 
    'microscope_controller': 18,
    'microscope_eye': 19,
    'microscope_screen': 20,
    'needle': 21,
    'operating_room': 22,
    'operating_table': 23,
    'operation_table': 24,
    'patient': 25,
    'scalpel': 26,
    'scissors': 27,
    'syringe': 28,
    'tissue_mark': 29,
    'tissue_paper': 30,
    'ultrasound_gel': 31,
    'ultrasound_machine': 32,
    'ultrasound_probe': 33,
    'ultrasound_screen': 34,
    'unsterile_instruments': 35,
    'vertebrae': 36
}

RELATION_VOCAB = {
    'anesthasing': 0,
    'applying': 1,
    'aspirating': 2,
    'checking': 3,
    'closeTo': 4,
    'controlling': 5,
    'cutting': 6,
    'disinfection': 7,
    'dressing': 8,
    'dropping': 9,
    'entering': 10,
    'holding': 11,
    'injecting': 12,
    'inserting': 13,
    'looking': 14,
    'lyingOn': 15,
    'manipulating': 16,
    'positioning': 17,
    'preparing': 18,
    'removing': 19,
    'scanning': 20,
    'touching': 21,
    'wearing': 22
}


scene_graph_name_to_vocab_idx = {
    # Entities
    'anesthetist': 0,
    'anesthesia_equipment': 1,
    'antiseptic': 2,
    'assistant': 3,
    'bin': 4,
    'body_marker': 5,
    'circulator': 6,
    'cotton': 7,
    'curette': 8,
    'dressing_material': 9,
    'forceps': 10,
    'gloves': 11,
    'head_surgeon': 12,
    'health_monitor': 13,
    'herbal_disk': 14,
    'instrument_table': 15,
    'instruments': 16,
    'microscope': 17,
    'microscope_controller': 18,
    'microscope_eye': 19,
    'microscope_screen': 20,
    'needle': 21,
    'operating_room': 22,
    'operating_table': 23,
    'patient': 24,
    'scalpel': 25,
    'scissors': 26,
    'syringe': 27,
    'tissue_mark': 28,
    'tissue_paper': 29,
    'ultrasound_gel': 30,
    'ultrasound_machine': 31,
    'ultrasound_probe': 32,
    'ultrasound_screen': 33,
    'unsterile_instruments': 34,
    'vertebrae': 35,
    # Relations (start at ID 36)
    'anaesthetising': 36,  # synonym
    'applying': 37,
    'aspirating': 38,
    'looking': 39,  # synonym
    'closeto': 40,  # synonym
    'controlling': 41,
    'cutting': 42,
    'disinfection': 43,
    'dressing': 44,
    'dropping': 45,
    'entering': 46,
    'holding': 47,
    'injecting': 48,
    'inserting': 49,
    'lyingon': 50,  # synonym
    'manipulating': 51,
    'positioning': 52,
    'preparing': 53,
    'removing': 54,
    'scanning': 55,
    'touching': 56,
    'wearing': 57
}
vocab_idx_to_scene_graph_name = {v: k for k, v in scene_graph_name_to_vocab_idx.items()}

# Define source groups
EGO_SOURCES = {"assistant", "head_surgeon", "circulator", "anesthetist"}
EXO_SOURCES = {"or_light", "microscope", "simstation"}
ROBOT_SOURCES = {"ultrasound"}
EXTERNAL_PATTERN = re.compile(r"external_[1-5]")

GAZE_FIXATION = {
    "x" : -9, 
    "y" : -4, 
}

GAZE_FIXATION_TO_TAKE = {
    "assistant" : [
        "data/Ultrasound/1/take/1",
        "data/Ultrasound/2/take/1",
        "data/Ultrasound/2/take/2",
        "data/Ultrasound/2/take/3",
        "data/Ultrasound/2/take/4",
        "data/Ultrasound/2/take/5",
        "data/Ultrasound/2/take/6",
    ],
    "head_surgeon" : [
        "data/Ultrasound/3/take/1",
        "data/Ultrasound/3/take/2",
        "data/Ultrasound/3/take/3",
        "data/Ultrasound/3/take/4",
        "data/Ultrasound/4/take/1",
        "data/Ultrasound/4/take/2",
        "data/Ultrasound/4/take/3",
        "data/Ultrasound/4/take/4",
        "data/Ultrasound/4/take/5",
        "data/Ultrasound/4/take/6",
    ],
    "anesthetist" : [
        "data/MISS/1/take/1",
        "data/MISS/1/take/2",
        "data/MISS/1/take/3",
        "data/MISS/2/take/1",
        "data/MISS/2/take/2",
        "data/MISS/2/take/3",
        "data/MISS/2/take/4",
        "data/MISS/3/take/1",
        "data/MISS/3/take/2",
        "data/MISS/3/take/3",
    ],
    "circulator" :
    [
        "data/MISS/1/take/4",
        "data/MISS/3/take/4",
        "data/MISS/3/take/5",
        "data/MISS/3/take/6",
        "data/MISS/4/take/1",
        "data/MISS/4/take/2",
    ]

}

entity_synonyms = {
    "operating_table" : ["operation_table", "operating_table"],
    'anesthetist': ['anaesthetist'],
    "microscope": ["microcope"]
}

relation_synonyms = {
    "closeto" : ["closeTo"],
    "lyingon" : ["lyingOn"],
    "looking" : ["looking", "checking"],
    "anaesthetising" : ["anesthasing"]
}

# Reverse synonym mapping
# Adopted from https://github.com/egeozsoy/MM-OR/scene_graph_generation/scene_graph_prediction/scene_graph_helpers/dataset/dataset_utils.py#L57
def reverse_synonym_mapping(synonyms_dict):
    reversed_dict = {}
    for key, synonyms_list in synonyms_dict.items():
        for synonym in synonyms_list:
            reversed_dict[synonym] = key
    return reversed_dict


def get_reversed_dict(sources_dict):
    reversed_dict = {}
    for key, value in sources_dict.items():
        reversed_dict[value] = key
    return reversed_dict

# Applying the function
reversed_entity_synonyms = reverse_synonym_mapping(entity_synonyms)
reversed_relation_synonyms = reverse_synonym_mapping(relation_synonyms)
reversed_sources = get_reversed_dict(SOURCES)

def map_scene_graph_name_to_vocab_idx(name):
    name = name.lower()
    # Synonym mapping
    if name in reversed_relation_synonyms:
        name = reversed_relation_synonyms[name]

    if name in reversed_entity_synonyms:
        name = reversed_entity_synonyms[name]
    return scene_graph_name_to_vocab_idx[name]

# Adopted from https://github.com/egeozsoy/MM-OR/scene_graph_generation/scene_graph_prediction/scene_graph_helpers/dataset/dataset_utils.py#L79
def map_vocab_idx_to_scene_graph_name(vocab_idx):
    return vocab_idx_to_scene_graph_name[vocab_idx]
