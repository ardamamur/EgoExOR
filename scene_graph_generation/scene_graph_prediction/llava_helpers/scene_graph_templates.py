"""
Direct copies of MM-OR's (https://github.com/egeozsoy/MM-OR) prompt templates for exact alignment.
"""

# Main scene graph template from MM-OR's generate_dataset_format_for_llava.py but with EgoExOR's vocabulary
SCENE_GRAPH_PROMPT = 'Entities: [assistant, anaesthetist, circulator, head surgeon, operating table, operating room, instrument table, antiseptic, anesthesia equipment, bin, body marker, cotton, gloves, health monitor, instruments, microscope controller, microscope eye, microscope screen, needle, patient, scalpel, scissors, forceps, curette, tissue mark, tissue paper, ultrasound gel, ultrasound machine, ultrasound probe, ultrasound screen, unsterile instruments, vertebrae, herbal disk, dressing material]. Predicates: [disinfection, close to, touching, cutting, holding, lying on, manipulating, preparing, scanning, positioning, controlling, looking, inserting, injecting, dropping, applying, dressing, entering, removing, aspirating, wearing, anaesthetising]. Given the following scene graph memory representation, generate a scene graph for timepoint T. The output should strictly be a list of triplets, each in the format "entity1,entity2,predicate;". Do not provide a narrative or descriptive text.'

# Memory and metadata enhancement patterns from MM-OR
MEMORY_PREFIX = '<memory_start>: '
MEMORY_SUFFIX = '<memory_end>.'
SPEECH_PREFIX = '<speech_transcript_start>: '
SPEECH_SUFFIX = ' <speech_transcript_end>. '

# Scene graph formatting from MM-OR
SCENE_GRAPH_PREFIX = '<SG> '
SCENE_GRAPH_SUFFIX = ' </SG>' 