# Technical Report: Automated CAPTCHA Recognition for Fudan Sports Reservation

## Abstract

This report describes the CAPTCHA recognition module used in an automated Fudan University sports-venue reservation workflow. The system is designed for a click-based CAPTCHA where the user must identify a four-character idiom and click the corresponding characters in the correct order. The implementation combines browser automation, image preprocessing, foreground segmentation, a lightweight single-character neural classifier, and a constrained idiom-level matching step.

The resulting pipeline runs on CPU, requires no GPU during inference, and is integrated into a scheduled reservation tool. In internal tests, recognition, coordinate recovery, clicking, and submission typically complete within two seconds, with an end-to-end recognition-click-submit success rate above 90 percent under the tested website conditions.

## 1. Problem Setting

The reservation website introduces a click-based CAPTCHA before a booking request can be submitted. A typical interaction requires the user to:

1. Open the reservation page for a target venue and time slot.
2. Trigger the verification challenge.
3. Read the CAPTCHA image.
4. Identify the requested idiom or character sequence.
5. Click the four target characters in the required order.
6. Submit the reservation request.

The goal of this project is to automate this workflow after the user has configured the target venue, date, and time. The CAPTCHA module is one component of a larger reservation system that also handles page navigation, time-slot selection, scheduled execution, retries, logs, and screenshot-based debugging.

## 2. Design Goals

The CAPTCHA recognizer was designed around four practical constraints:

- **End-to-end latency:** recognition and clicking should finish fast enough for time-sensitive reservation windows, usually under two seconds.
- **CPU-only deployment:** the recognizer should run on a normal laptop or lightweight server without GPU inference.
- **Recoverability:** occasional failures are acceptable if the system can refresh, retry, and preserve debugging evidence.
- **Coordinate fidelity:** predicted characters must be mapped back to browser click positions accurately enough to pass the website challenge.

## 3. Data Collection and Labeling

Several thousand CAPTCHA images were collected from the university reservation website during development. The dataset covers the visual patterns observed in the production challenge, including:

- Chinese character glyphs appearing in idiom-based challenges.
- Distracting backgrounds and texture patterns.
- Character location variation within the CAPTCHA image.
- Moderate image noise, scaling, and browser-rendering differences.

The labeling process focused on single-character recognition. Rather than training a model to classify an entire CAPTCHA image directly, the pipeline first segments the image into character crops and then classifies each crop independently. This design reduces the output space and allows the final idiom-level decision to use both neural predictions and symbolic constraints.

## 4. Recognition Pipeline

The pipeline consists of six stages:

1. Capture the CAPTCHA region from the browser.
2. Suppress the background and isolate foreground character pixels.
3. Segment the image into candidate character regions.
4. Classify each character crop with a lightweight neural network.
5. Search over the idiom dictionary to find the most plausible ordered character sequence.
6. Convert the selected character regions into browser click coordinates and submit the challenge.

### 4.1 Browser Capture

The automation script uses browser control to locate the CAPTCHA element and capture the corresponding image region. In practice, full-window screenshots and element coordinates are used together to reduce errors caused by device-pixel-ratio differences, browser zoom, and window scaling.

The captured image is then normalized to a fixed working size before segmentation and classification. This gives the downstream image-processing steps a stable coordinate system.

### 4.2 Background Suppression

The CAPTCHA includes recurring background textures. The implementation stores representative background templates and compares the current CAPTCHA image against them. The closest background template is resized to match the current CAPTCHA and subtracted from the image.

This step converts the problem from recognizing characters in a textured image to identifying foreground regions that differ from the selected background. It is intentionally lightweight and deterministic, which helps keep inference latency low.

### 4.3 Character Segmentation

After background suppression, foreground pixels are clustered to identify candidate character regions. The current implementation uses density-based clustering over foreground coordinates, then computes a center for each detected region.

Each candidate region is cropped around its center and resized to a fixed input size for classification. The centers are preserved because they are later mapped back to click coordinates in the browser.

This segmentation step is a key engineering component. The model is only responsible for recognizing isolated character crops; it is not required to localize characters directly from the full CAPTCHA image.

### 4.4 Single-Character Classifier

The character recognizer is a lightweight neural classifier based on MobileNetV2. It predicts a distribution over the character vocabulary used in the idiom CAPTCHA set.

The model is small enough for CPU inference and is applied independently to the segmented character crops. For each crop, the classifier returns ranked candidate characters and confidence scores. These scores are passed to the idiom-level matching stage instead of being used greedily one character at a time.

### 4.5 Idiom-Level Matching

The CAPTCHA is not an arbitrary sequence of independent characters. It is drawn from a finite idiom set. The pipeline uses this structure to improve robustness:

- For each candidate idiom, the system checks whether its characters are plausible matches for the detected crops.
- It uses classifier confidence scores to assign idiom characters to segmented regions.
- It enforces that different idiom characters are mapped to different visual regions.
- It selects the idiom and click order with the highest aggregate score.

This constrained matching step is important because a single-character classifier may confuse visually similar Chinese characters. The idiom dictionary acts as a language-level prior that converts independent character predictions into a consistent four-character solution.

### 4.6 Coordinate Mapping and Clicking

Once the best idiom and region assignment are selected, the system maps each character center from the normalized CAPTCHA coordinate system back into the browser page coordinate system. Browser automation then clicks the characters in the predicted order and submits the challenge.

The implementation records screenshots and logs on failure so that coordinate errors, segmentation failures, and recognition mistakes can be debugged after a run.

## 5. Integration with Automated Reservation

The CAPTCHA recognizer is integrated into a full sports-reservation workflow:

1. The user configures the target campus, venue, date, and time slot.
2. The automation opens the reservation website and navigates to the target resource.
3. The script waits for the configured booking window or scheduled execution time.
4. It selects the desired venue and time slot.
5. It triggers the CAPTCHA challenge.
6. The recognition module solves the CAPTCHA and performs the click sequence.
7. The script submits the booking request.
8. If a run fails, the system can retry and preserve diagnostic logs/screenshots.

This integration is the main difference between a standalone CAPTCHA recognizer and the project as a complete automation system. The recognizer must be fast enough and stable enough to operate inside the time-sensitive reservation loop.

## 6. Evaluation

The main evaluation target is end-to-end operational success rather than isolated classification accuracy. The relevant outcome is whether the system can recognize the challenge, click the required characters, and submit the reservation request correctly.

The primary metrics used during development were:

- **Recognition-click-submit success rate:** whether the CAPTCHA challenge is passed and the booking request proceeds.
- **End-to-end latency:** time from CAPTCHA capture to completed click sequence.
- **Failure recoverability:** whether failed attempts produce useful logs and screenshots for debugging.

Under the tested conditions, the CPU-only CAPTCHA pipeline typically completes recognition and clicking in under two seconds and achieves over 90 percent end-to-end recognition-click-submit success.

## 7. Limitations

The system is engineering-oriented and depends on the observed CAPTCHA distribution. Its main limitations are:

- **Website changes:** changes to the CAPTCHA style, HTML layout, browser scaling, or reservation flow may require recalibration.
- **Segmentation errors:** if the foreground extraction or clustering step fails, the classifier receives poor crops.
- **Visual ambiguity:** visually similar characters can still be confused by the single-character classifier.
- **Operational dependency:** network latency and page-loading behavior can affect the broader reservation workflow even if CAPTCHA recognition succeeds.

For these reasons, the implementation includes retries and diagnostic artifacts rather than assuming a perfectly reliable one-shot pipeline.

## 8. Responsible Use

This project is intended as a personal engineering and automation exercise. It should be used responsibly and in compliance with applicable university policies, website rules, and fair-use expectations. No credentials or private account information are included in this report.

## 9. Summary

The project demonstrates a practical CAPTCHA-solving module embedded in a complete sports-reservation automation workflow. The key design choice is to split the problem into foreground segmentation, lightweight single-character recognition, and idiom-constrained sequence matching. This makes the system fast enough for CPU-only scheduled use while maintaining a high operational success rate under the tested conditions.
