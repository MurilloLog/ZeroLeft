# ZeroLeft: An Intelligent Platform for Surgical Tool Traceability Based on Meta SAM 3

[![Meta Global AI Developer Hackathon](https://img.shields.io/badge/Meta_Global_AI_Hackathon-2026-blue.svg)](https://cerebralvalley.ai/events/~/e/meta-ai-developer-hackathon?modalCloseUrl=%2Fevents%2Fregistered)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org/)

<!--![ZeroLeft Demo Banner](docs/demo_banner.png) -->

Surgical instrument traceability remains a significant challenge in modern healthcare, where counting, identification, and verification processes continue to depend heavily on manual workflows. Although recent advances in computer vision have shown promising results, real-world surgical environments remain particularly challenging due to reflections, occlusions, overlapping instruments, and visually similar tool geometries.

ZeroLeft is an experimental platform developed to explore the potential of Meta SAM 3 for surgical instrument segmentation and traceability-related tasks. Built for the Meta Global AI Developer Hackathon 2026, the project investigates whether foundation vision models can provide a flexible alternative to traditional task-specific approaches, while serving as a testbed for future research in intelligent surgical instrument management.

To support this evaluation, ZeroLeft incorporates the following capabilities:

- **Zero-Shot Generalization**: Leverages Meta SAM 3.1 to isolate fine-grained surgical tools (forceps, scalpels, clamps) and anatomical regions without custom fine-tuning.
- **Interactive Spatial Prompting**: Enables instrument segmentation through simple point, box, or text prompts, facilitating rapid experimentation in complex scenarios.
- **Automated Spatial Reporting**: Integrates Muse Spark to analyze mask dimensions, spatial orientation, and contextual information derived from segmented instruments.

This repository contains the implementation, experiments, and supporting resources used to assess the opportunities and limitations of foundation models as a step toward future surgical instrument traceability systems.

## Dataset & Testing
ZeroLeft was evaluated using open-access surgical images from [Surgical Tools](https://www.kaggle.com/datasets/jocelyndumlao/surgical-tools/data) and [HOSPI-Tools Dataset](https://zenodo.org/records/6796559?preview_file=Hospi_Tools_small_Mask_RCNN.zip). The framework processes open-surgery frames, applying zero-shot boundary isolation under complex conditions like specular reflections, blood, and motion blur.

## Roadmap & Future Work
[In proccess] Zero-shot single-frame point-prompt segmentation via SAM 3.1.

[ ] Multi-prompt bounding box isolation.

[ ] Real-time video stream processing and temporal mask tracking.

[ ] Integration with WebXR/Augmented Reality viewports for heads-up surgical displays.

## License
Distributed under the MIT License. See [LICENSE](./LICENSE) for more information.

## Contact & Team
Developed for the Meta Global AI Developer Hackathon 2026 (Indie Developer Track).

Gustavo Adolfo Murillo | [LinkedIn](https://www.linkedin.com/in/murillolog/)