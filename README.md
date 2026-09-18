<div align="center">
<img src="./assets/project-cover.png" alt="Student Performance Data Science" width="100%" />

# Student Performance Data Science

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" /> <img src="https://img.shields.io/badge/Machine%20Learning-0f172a?style=for-the-badge&logo=scikit-learn&logoColor=F7931E" alt="Machine Learning" /> <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />

</div>

---


مشروع علم بيانات قابل لإعادة الإنتاج لتحليل أداء طلاب المرحلة الثانوية وبناء مهمتي **الانحدار** للتنبؤ بالدرجة النهائية و**التصنيف** إلى Low/Medium/High. يتضمن المشروع جمع البيانات، التنظيف، EDA، اختبارات إحصائية، هندسة ميزات، مقارنة نماذج، GridSearch، تحليل أخطاء، SHAP، Dashboard، API، اختبارات، وتوثيق أكاديمي.

## البيانات والحدود

تستخدم النسخة الحالية ملف الرياضيات من مجموعة **UCI Student Performance**. البيانات عامة ومجهولة الهوية. لا تستخدم `G1` أو `G2` كمدخلات عند التنبؤ بـ`G3` لتقليل تسرب المعلومات. مستويات الأداء معرفة كالتالي: Low للدرجات 0–9، Medium للدرجات 10–14، وHigh للدرجات 15–20. هذه الحدود تعليمية وليست سياسة مؤسسية.

## التشغيل الكامل

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train.py
python scripts/run_analysis.py
pytest -q
streamlit run app/streamlit_app.py
```

### Windows PowerShell

الأوامر `source .venv/bin/activate` خاصة بـLinux/macOS ولا تعمل في PowerShell. يجب أولًا فتح المجلد الذي يحتوي مباشرة على `requirements.txt` و`src` و`scripts`، وليس المجلد الأب الذي يحتوي مجلد المستودع.

```powershell
cd "E:\الجامعه\level 3\level 3 term 1\علم بيانات\عملي\project trm 1\student-performance-data-science-main"
dir requirements.txt, scripts, src
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts\train.py
python scripts\run_analysis.py
python -m pytest -q
python -m streamlit run app\streamlit_app.py
```

يمكن تنفيذ الإعداد والتحقق تلقائيًا من جذر المستودع عبر:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_windows.ps1
```

يتطلب المشروع Python 3.10 أو أحدث، ويوصى باستخدام Python 3.12. إذا كان الأمر `py -3.12` غير متاح، ثبّت Python من [المصدر الرسمي](https://www.python.org/downloads/windows/) ثم أعد فتح PowerShell.

إذا ظهر `requirements.txt` أو `scripts\train.py` غير موجود، فأنت في المجلد الخطأ. نفّذ `dir` وتأكد من ظهور هذه الملفات قبل متابعة الأوامر. وإذا ظهر `ModuleNotFoundError` مثل `pandas` أو `fastapi`، فهذا يعني أن التثبيت لم يتم في البيئة الحالية؛ استخدم `python -m pip install -r requirements.txt` بعد تفعيل `.venv`.

ينشئ `train.py` البيانات المعالجة والنماذج الأساسية. ينشئ `run_analysis.py` رسوم EDA واختبارات إحصائية ونتائج GridSearch وتحليل الأخطاء وملفات SHAP. جميع النتائج تحفظ في `data/processed/` والرسوم في `reports/figures/`.

## API

بعد تثبيت الاعتمادات وتشغيل التدريب:

```bash
./scripts/run_api.sh
```

النقاط المتاحة هي `GET /health` و`POST /predict`. توثيق OpenAPI يظهر في `/docs`. لا تستخدم الواجهة لاتخاذ قرارات عالية الأثر بشأن الطلاب.

## بنية المستودع

- `src/data_pipeline.py`: التنزيل والتنظيف وبناء الميزات.
- `src/modeling.py`: Pipelines والتدريب والتقييم الأساسي.
- `src/analysis.py`: EDA، الإحصاء، GridSearch، تحليل الأخطاء، permutation importance وSHAP.
- `src/predict.py`: التحقق من المدخلات والتنبؤ.
- `src/api.py`: واجهة FastAPI.
- `app/streamlit_app.py`: صفحات Overview وExploration وPrediction وExplainability وEvaluation.
- `notebooks/`: الدفاتر الثمانية المطلوبة وفق التسلسل الأكاديمي.
- `tests/`: اختبارات المعالجة، الميزات، النموذج، التنبؤ، والتحقق.
- `docs/data_dictionary.md`: قاموس البيانات.
- `reports/report.md`: التقرير الأكاديمي الكامل.
- `reports/presentation.md`: مخطط العرض النهائي.

## النتائج الفعلية

في التشغيل الحالي على 395 سجلًا و34 عمودًا مشتقًا، لم توجد قيم مفقودة أو صفوف مكررة. حقق النموذج الأولي Random Forest للانحدار RMSE = 4.369 وMAE = 3.326 وR² = 0.232. بعد الضبط، بلغ RMSE = 4.411 وMAE = 3.352. لذلك لم يُستبدل النموذج الأولي في واجهة التنبؤ لأن معيار الانحدار الأساسي كان أفضل.

في التصنيف، حقق Random Forest المضبوط Accuracy = 0.557 وMacro F1 = 0.497، متفوقًا على النموذج الأولي الذي حقق Accuracy = 0.506 وMacro F1 = 0.396. من المهم قراءة Macro F1 مع مصفوفة الالتباس لأن الفئات غير متوازنة.

أظهرت permutation importance أن `absences` و`failures` كانتا الأعلى في هذا النموذج. إحصائيًا، ارتبطت `failures` بالدرجة النهائية ارتباطًا سلبيًا دالًا، بينما لم يظهر ارتباط بيرسون للحضور دلالة عند α = 0.05. أظهر اختبار Welch فرقًا دالًا في الدرجات بين مجموعتي نية التعليم العالي. هذه علاقات إحصائية وليست إثباتًا للسببية.

## المصدر والتوثيق

[1]: https://archive.ics.uci.edu/dataset/320/student+performance "UCI Student Performance Dataset"
[2]: https://scikit-learn.org/stable/modules/compose.html "Scikit-learn Pipelines and composite estimators"
[3]: https://shap.readthedocs.io/en/latest/ "SHAP documentation"

## الترخيص

هذا المشروع مرخص وفق MIT. يرجى مراجعة شروط مصدر البيانات قبل إعادة التوزيع.
