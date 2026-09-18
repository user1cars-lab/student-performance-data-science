# Student Performance Data Science

مشروع علم بيانات قابل لإعادة الإنتاج لتحليل أداء طلاب المرحلة الثانوية وبناء مهمتي **الانحدار** للتنبؤ بالدرجة النهائية و**التصنيف** إلى Low/Medium/High. يتضمن المشروع خط معالجة بيانات، مقارنة نماذج، تقييمًا عبر cross-validation، واختبارات وواجهة Streamlit.

## نطاق النسخة الأولى

تستخدم النسخة الحالية ملف الرياضيات من مجموعة **UCI Student Performance**. لا تستخدم `G1` أو `G2` كمدخلات عند التنبؤ بـ`G3` لأنهما درجات مرحلية قد تسبب تسربًا أو لا تكون متاحة في وقت التدخل. مستويات الأداء معرفة كالتالي: Low للدرجات 0–9، Medium للدرجات 10–14، وHigh للدرجات 15–20.

## التشغيل

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train.py
pytest -q
streamlit run app/streamlit_app.py
```

يقوم `scripts/train.py` بتنزيل المصدر الرسمي إلى `data/raw/`، تنظيفه، حفظ البيانات المعالجة، تدريب نماذج baseline وRidge وRandom Forest وGradient Boosting للانحدار، وتدريب baseline وLogistic Regression وRandom Forest للتصنيف. تحفظ النماذج المختارة في `models/`.

## بنية المستودع

- `src/data_pipeline.py`: التنزيل والتنظيف وبناء الميزات.
- `src/modeling.py`: preprocessing داخل Pipeline، التدريب والتقييم والحفظ.
- `src/predict.py`: التحقق من المدخلات والتنبؤ.
- `app/streamlit_app.py`: Dashboard أولية للتحليل والتنبؤ.
- `tests/`: اختبارات تمنع regressions وتتحقق من منع التسرب.
- `docs/data_dictionary.md`: قاموس البيانات والافتراضات.
- `reports/report.md`: التقرير الأكاديمي الأولي وخطة القياس.

## التقييم والتفسير

تستخدم مهمة الانحدار MAE وRMSE وR²، مع RMSE عبر 5-fold cross-validation. تستخدم مهمة التصنيف Accuracy وMacro Precision وMacro Recall وMacro F1. ستضاف رسوم EDA وSHAP في المرحلة التالية بعد تثبيت نتائج التشغيل. التفسير يصف سلوك النموذج ولا يثبت علاقة سببية.

## القيود والأخلاقيات

البيانات تعليمية عامة ومجهولة الهوية، ولا يجوز استخدام النموذج لاتخاذ قرارات عالية الأثر بشأن الطلاب. التعميم محدود بسياق المدارس والبلدان والوقت الذي جُمعت فيه البيانات. كما أن حدود الأداء المستخدمة للتصنيف افتراضات توضيحية وليست معيارًا رسميًا.

## المصدر

[1]: https://archive.ics.uci.edu/dataset/320/student+performance "UCI Student Performance Dataset"

## الترخيص

هذا المشروع مرخص وفق MIT. يرجى مراجعة شروط مصدر البيانات قبل إعادة التوزيع.
