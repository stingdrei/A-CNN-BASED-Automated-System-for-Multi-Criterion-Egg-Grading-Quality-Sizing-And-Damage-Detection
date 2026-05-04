Automated Egg Detection and Grading System

4LOOP

Asis, Benaiah        3A
Ballarta, Joan       3A
Jugado, Michael Andrei    3C
Lomugdang, Lui Franz      3C

Project Description

The goal of this project is to develop an efficient, automated egg detection and grading system that utilizes deep learning to modernize traditional agricultural sorting processes. Its primary objective is to replace inconsistent manual labor with a consistent, efficient, and user-friendly digital solution. By leveraging computer vision, the system can identify eggs in images and live video feeds and classify them by damage status as either damaged or not damaged. Size and weight estimation features are currently under development and will be integrated in a subsequent update.

The system is built using the YOLOv8s model, a lightweight object detection architecture fine-tuned on a custom egg dataset. It is deployed through a full-stack web application comprising a FastAPI backend and a React frontend, as well as a standalone desktop live-view script. Both deployment modes are capable of delivering results on standard hardware without requiring a dedicated graphics processing unit.

Target Audience

The system is primarily designed for local poultry farmers and egg distributors who require a cost-effective method to standardize quality control without the expense of industrial-scale machinery. By providing an accessible web-based tool that operates on standard computing hardware, it also serves small-scale agricultural entrepreneurs seeking to transition from manual to digital sorting processes. Additionally, the project provides a reproducible framework for agricultural technology researchers interested in the practical application of computer vision in food production and quality assurance.

Technical Specifications

The system is built on a modern full-stack architecture that combines deep learning, RESTful application programming interfaces, and a responsive web interface.

Software Stack

Primary Language: Python for backend and machine learning components, TypeScript for the frontend
Deep Learning Framework: PyTorch for model training, inference, and optimization
Core Model: YOLOv8s fine-tuned for egg detection and damage classification
Computer Vision: OpenCV for image processing, annotation generation, and live video capture
Backend Framework: FastAPI with SQLAlchemy object-relational mapping and PostgreSQL database
Frontend Framework: React 18 with TypeScript, Vite build tool, and Tailwind CSS styling
Authentication: JSON Web Tokens using python-jose and passlib with bcrypt hashing
Numerical Processing: NumPy for array operations and mathematical computations
State Management: Zustand for frontend application state
Development Environment: Visual Studio Code

Hardware Requirements

Input: Webcam for live capture, video files, or image uploads in JPG, PNG, WebP, or BMP formats
Processing: Standard computer or laptop. A graphics processing unit is optional but recommended for faster model training. CPU-based inference operates at approximately 28 to 30 frames per second.

System Workflow

The system operates through a structured pipeline that transforms raw visual input into a graded classification output.

Image Acquisition: Input is captured through a live camera feed, loaded from a video file, or uploaded via the web interface.
Object Detection: The YOLOv8s model processes each frame to identify individual eggs and generates bounding boxes with associated confidence scores.
Confidence Filtering: Detections falling below the configured confidence threshold, set to 0.75 by default, are discarded to minimize false positive rates.
Size and Weight Estimation: This feature is currently under development. The planned implementation will convert bounding box dimensions to real-world measurements using a camera calibration factor expressed in millimeters per pixel, with eggs categorized by size and weight estimated using an empirical formula.
Grade Assignment: The current release classifies eggs strictly by damage status. Damaged eggs are marked as Reject and intact eggs are marked as Pass. The extended grading scheme that incorporates size categories to produce AA, A, or B grades for intact eggs is planned for a future update.
Output Display: Results are presented through an annotated image with color-coded bounding boxes, a detailed detection table, and a dashboard displaying grade distribution statistics across all user predictions.

Development Timeline

The project followed an iterative development plan spanning four weeks.

Phase 1: Project Setup and Data Collection. Project initialization, conversion of the dataset to YOLO annotation format, and training of a convolutional neural network for preliminary damage detection.

Phase 2: YOLO Integration. Training of the YOLOv8s model for 50 epochs, implementation of live detection with multi-egg support, and integration of centroid-based object tracking to prevent duplicate counting across video frames.

Phase 3: System Enhancements. Comprehensive improvements including confidence threshold optimization, camera calibration tool development, CSV logging for detection statistics, and creation of system architecture documentation.

Phase 4: Full-Stack Web Application. Development of the FastAPI backend with PostgreSQL database integration, JSON Web Token authentication, and the React frontend featuring user dashboard, image upload interface, and prediction history management.

The development followed a four-phase iterative plan, progressing from initial data collection and model training to system refinement and full-stack deployment. During the first week, the team finalized the dataset and trained the initial damage classification model. By the second week, YOLOv8s was integrated and fine-tuned, achieving a substantial improvement in precision from 58.32 percent to 73.99 percent and in recall from 5.62 percent to 75.00 percent. The third week focused on system optimizations including confidence threshold tuning and object tracking implementation. In the fourth week, the team constructed the complete web application with user authentication, prediction history, and dashboard analytics. The project has successfully delivered a functional minimum viable product with both a desktop live-view tool and a web-based interface.

Current Status and Finalized MVP Features

The project has transitioned from a basic prototype to a refined minimum viable product with both desktop and web deployment modes.

Key Achieved Milestones

Real-Time Detection: Automated identification of eggs from live camera feeds and video files at approximately 28 to 30 frames per second.
Automated Grading: Binary classification of eggs as damaged or not damaged based on shell condition and visible defects. Extended grading with size categories is under development.
Web Application: Full-stack platform featuring user authentication, image upload, prediction history tracking, and dashboard analytics.
Object Tracking: Centroid-based tracking algorithm that prevents duplicate counting of the same egg across consecutive video frames.
Camera Calibration: Pixel-to-millimeter conversion framework is implemented in the codebase and will be activated for size and weight estimation in the next release.
System Stability: Improved reliability through confidence threshold optimization and dataset augmentation exceeding 1500 images.

Current Capabilities and Finalized Features

The finalized minimum viable product provides real-time egg detection from live camera input and binary classification based on shell condition. Eggs are identified as either damaged or not damaged. Size and weight estimation features are structurally implemented in the codebase but are not yet active in the current release and will be enabled in a subsequent update. The system offers a web-based interface for image upload and analysis, complete with annotated output images, detailed detection tables, and grade distribution charts. The desktop live-view mode supports keyboard controls for real-time confidence adjustment, video pausing, and screenshot capture. All detections are logged to comma-separated values files for subsequent statistical analysis.

Current testing demonstrates 73.99 percent precision, 75.00 percent recall, and 81.46 percent mean average precision at an intersection over union threshold of 50 percent. The system maintains consistent performance at 28 to 30 frames per second on standard central processing unit hardware. Technical adjustments including confidence threshold tuning and dataset augmentation have reduced the false positive rate from approximately 18 percent to 12 percent, ensuring efficient and reliable operation without specialized hardware.

Conclusion

The system demonstrates that deep learning can serve as an accessible and effective tool for the agricultural sector. By bridging the gap between advanced computer vision technology and practical farming requirements, this project provides a scalable foundation for future enhancements. The immediate next development cycle will activate the size and weight estimation pipeline, enabling the full multi-criterion grading scheme. Further planned improvements include cloud-based graphics processing, expanded datasets with finer damage severity classifications, and multi-camera deployment configurations. The project validates the feasibility of automated egg grading and offers a practical pathway toward more efficient and reliable quality control in egg production.
