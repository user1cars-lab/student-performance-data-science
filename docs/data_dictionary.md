# Data Dictionary

المصدر هو **UCI Student Performance**. تستخدم النسخة الحالية ملف الرياضيات `student-mat.csv` الذي يحتوي على 395 سجلًا وخصائص ديموغرافية ومدرسية واجتماعية ودرجات مرحلية ونهائية.

| Variable | Type | Description | Used for prediction |
|---|---|---|---|
| school | categorical | School identifier | Yes |
| sex | categorical | Student sex | Yes |
| age | integer | Student age | Yes |
| address | categorical | Urban/rural residence | Yes |
| studytime | ordinal | Weekly study-time category | Yes |
| failures | integer | Number of past class failures | Yes |
| absences | integer | Number of school absences | Yes |
| G1, G2 | numeric | First and second period grades | **No: excluded to reduce leakage** |
| G3 | numeric | Final grade on a 0–20 scale | Regression target |
| performance_level | categorical | Low (0–9), Medium (10–14), High (15–20) | Classification target |
| study_engagement | numeric | `studytime × higher education intention` | Derived feature |

The remaining variables follow the names and definitions in the official UCI metadata. The labels are project-defined thresholds and should not be treated as institutional policy.
