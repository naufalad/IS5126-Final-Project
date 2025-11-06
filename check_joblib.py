import os, sys, joblib

BASE = os.path.dirname(os.path.abspath(__file__))
# 确保能 import 到 deployment/app/email_pipeline.py 作为顶层模块 email_pipeline
sys.path.insert(0, os.path.join(BASE, "deployment", "app"))

p = r"E:\Learning\NUS\Sem 1\CS5126 Handas-on with Applied Analytics\IS5126-Final-Project\IS5126-Final-Project\deployment\models\chocka.joblib"
m = joblib.load(p)

print(type(m))
print([a for a in dir(m) if a in ("predict","predict_proba","transform","fit","classes_","steps")])