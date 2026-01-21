# Take-Level Procedural Metadata

This document provides **take-level metadata** for the EgoExOR dataset, describing the **procedural context, clinical phase, and scripted anomalies** for each recorded take.

The goal of this metadata is to support:
- phase-aware surgical workflow modeling,
- anomaly-aware learning and evaluation,
- multimodal perception and reasoning in operating-room environments.

EgoExOR includes recordings from **two distinct procedures**:
1. **Ultrasound-Guided Injection (UI)**
2. **Minimally Invasive Spine Surgery (MISS)**

Both procedures are recorded using **egocentric and exocentric sensor setups**, as described in the EgoExOR paper.

---

## 1. General Principles

- Each **take** corresponds to a single, coherent procedural execution.
- Takes may follow a **nominal (no-anomaly) workflow** or include **scripted anomalies**.
- Scripted anomalies are clinically plausible, controlled, and isolated per take.
- Procedures are segmented into **clinically meaningful phases**.

---

## Ultrasound-Guided Injection (UI)

### Phase 1 – Patient Entry and Skin Antisepsis

**Take 1**
- Standard patient entry and antiseptic preparation  
- Ultrasound equipment present but not yet actively used  
- No anomalies

---

### Phase 2 – Patient Entry and Skin Antisepsis (Repeated)

**Take 1-2**
- Standard patient preparation and disinfection  
- Ultrasound equipment present but inactive  
- No anomalies

**Take 3-6**
- Patient preparation and disinfection  
- Ultrasound equipment present but inactive  
- Scripted anomaly: Instrument dropped

---

### Phase 3 – Ultrasound-Based Target Identification

**Take 1-2**
- Anatomical target localization using real-time ultrasound imaging  
- Direct patient contact without gloves  
- Scripted anomaly: Gloves omitted

**Take 3-4**
- Ultrasound-based target identification  
- External skin marker used to indicate target area  
- Procedural variation (no anomaly)

---

### Phase 4 – Ultrasound-Guided Injection and Cleanup (Set A)

**Take 1-2**
- Needle insertion and injection under real-time ultrasound guidance  
- Standard injection workflow  
- No anomalies

**Take 3-4**
- Ultrasound-guided injection attempt  
- Mechanical resistance during injection  
- Scripted anomaly: Syringe jammed

**Take 5-6**
- Ultrasound-guided needle advancement  
- Needle position corrected under ultrasound visualization  
- Scripted anomaly: Needle repositioning

---

### Phase 5 – Ultrasound-Guided Injection and Cleanup (Set B)

**Take 1-2**
- Ultrasound-guided injection and cleanup  
- Standard workflow  
- No anomalies

**Take 3-4**
- Ultrasound-guided needle adjustment  
- Corrective repositioning performed  
- Scripted anomaly: Needle repositioning

**Take 5-6**
- Ultrasound-guided injection attempt  
- Injection interrupted due to mechanical issue  
- Scripted anomaly: Syringe jammed
---

## Minimally Invasive Spine Surgery (MISS)

### Phase 1 – Patient Entry, Anesthesia, and Surgical Field Preparation

**Take 1-2**
- Patient entry and positioning  
- Anesthesia induction and sterile preparation  
- No anomalies

**Take 3**
- Patient prepared for surgery  
- Required surgical instrument unavailable  
- Scripted anomaly: Instrument missing

**Take 4**
- Patient preparation and anesthesia complete  
- Intra-team role reassignment performed  
- Scripted anomaly: Role changes (anesthetist ↔ circulator, head surgeon ↔ assistant)

---

### Phase 2 – Incision and Initial Surgical Access

**Take 1-2**
- Skin incision and initial access established  
- Surgical microscope correctly aligned  
- No anomalies

**Take 3-4**
- Initial access established  
- Suboptimal visualization due to microscope alignment  
- Scripted anomaly: Microscope misalignment

---

### Phase 3 – Microscope-Assisted Disc Removal

**Take 1-2**
- Disc removal under microscope guidance  
- Stable visualization and team coordination  
- No anomalies

**Take 3**
- Disc removal initiated  
- Visualization degraded due to microscope misalignment  
- Scripted anomaly: Microscope misalignment

**Take 4**
- Disc removal in progress  
- Temporary staff role reassignment  
- Scripted anomaly: Role changes (anesthetist ↔ circulator)

**Take 5-6**
- Disc removal interrupted  
- Additional instrument required and role reassignment  
- Scripted anomalies: Scalpel required, role changes

---

### Phase 4 – Post-Procedural Cleanup and Patient Dressing

**Take 1-2**
- Wound dressing and operative field cleanup  
- Preparation for patient transfer  
- No anomalies

---

## 4. Intended Use

This metadata enables:
- phase- and role-aware modeling,
- anomaly detection and anticipation,
- multimodal fusion across egocentric and exocentric views,
- benchmarking surgical workflow understanding methods.

---

## 5. Notes

- All anomalies are scripted but clinically realistic.
- Nominal takes are included in each phase as baselines.
- Terminology and structure align with the EgoExOR dataset paper.