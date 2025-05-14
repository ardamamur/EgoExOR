CAMERA_TYPE_MAPPING = {
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

GAZE_FIXATION = {
    "x" : -9, 
    "y" : -4, 
}

GAZE_FIXATION_TO_TAKE = {
    "assistant" : [
        "/data/Ultrasound/1/take/1",
        "/data/Ultrasound/2/take/1",
        "/data/Ultrasound/2/take/2",
        "/data/Ultrasound/2/take/3",
        "/data/Ultrasound/2/take/4",
        "/data/Ultrasound/2/take/5",
        "/data/Ultrasound/2/take/6",
    ],
    "head_surgeon" : [
        "/data/Ultrasound/3/take/1",
        "/data/Ultrasound/3/take/2",
        "/data/Ultrasound/3/take/3",
        "/data/Ultrasound/3/take/4",
        "/data/Ultrasound/4/take/1",
        "/data/Ultrasound/4/take/2",
        "/data/Ultrasound/4/take/3",
        "/data/Ultrasound/4/take/4",
        "/data/Ultrasound/4/take/5",
        "/data/Ultrasound/4/take/6",
    ],
    "anesthetist" : [
        "/data/MISS/1/take/1",
        "/data/MISS/1/take/2",
        "/data/MISS/1/take/3",
        "/data/MISS/2/take/1",
        "/data/MISS/2/take/2",
        "/data/MISS/2/take/3",
        "/data/MISS/2/take/4",
        "/data/MISS/3/take/1",
        "/data/MISS/3/take/2",
        "/data/MISS/3/take/3",
    ],
    "circulator" :
    [
        "/data/MISS/1/take/4",
        "/data/MISS/3/take/4",
        "/data/MISS/3/take/5",
        "/data/MISS/3/take/6",
        "/data/MISS/4/take/1",
        "/data/MISS/4/take/2",
    ]

}

EGOCENTRIC_SOURCES = {
    "head_surgeon",
    "assistant",
    "circulator",
    "anesthetist" 
}

EXOCENTRIC_SOURCES = {
    "or_light",
    "microscope",
    "external_1",
    "external_2",
    "external_3",
    "external_4",
    "external_5",
    "simstation",
    "ultrasound"
}