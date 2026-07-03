# Gold Price Forecasting in Thailand

A data science and machine learning project to analyze economic factors impacting gold prices in Thailand and compare the forecasting performance of statistical and machine learning models (Monthly Data, 2014 - 2023, 120 samples)

## Dataset & Variables
* **Target Variable:** Gold Bar Price in Thailand (GPT(THB))
* **Predictor Variables (Features):** Global Gold Price (GPW(USD)), Consumer Price Index (CPI), Policy Interest Rate (IR), Global Crude Oil Price (POIL(USD)), Exchange Rate (EXR)

---

## Methodology & Statistical Analysis

### 1. Feature Selection
Applied **Stepwise Regression** to select features with the highest statistical significance. The final model retains 3 key variables: **Global Gold Price (GPW(USD)), Exchange Rate (EXR), and Consumer Price Index (CPI)**

### 2. Statistical Assumption Checking
The final Linear Regression model strictly satisfies all classical linear regression assumptions:
* **Multicollinearity (VIF):** The VIF for all selected features ranges between 1.64 and 3.77, which is **well below the threshold of 5** (No multicollinearity issues)
* **Autocorrelation (Durbin-Watson):** The Durbin-Watson statistic is **2.186** (and 2.026 in the validation phase), confirming **no significant autocorrelation in the residuals**
* **Normality of Residuals:** The Shapiro-Wilk Test yielded a p-value of 0.537 (> 0.05), indicating that the residuals are **normally distributed**

---

## Model Evaluation & Performance

The forecasting models are evaluated and ranked by their **Mean Squared Error (MSE)** on the test set (from lowest to highest)

| Rank | Model | MSE (Mean Squared Error) |
| :---: | :--- | :---: |
| **1** | **Linear Regression (Best Model)** | **98,671.87** |
| 2 | XGBoost Regressor | 151,910.19 |
| 3 | Random Forest Regressor | 342,988.98 |
| 4 | Decision Tree Regressor | 739,270.83 |
| 5 | ARIMA (Time Series) | 49,236,132.33 |

### Conclusion
* **Best Model:** **Linear Regression** outperformed all other models, achieving the lowest MSE and an outstanding **R-squared of 0.994** (explaining 99.4% of the variance in Thai gold prices)
* **Key Insights:** By selecting features via Stepwise Regression and ensuring all statistical assumptions were met, the linear model accurately captured the strong, linear relationships between global gold prices, exchange rates, and local prices without overfitting Residual errors still persist due to the exclusion of external event-driven variables, such as wars, pandemics, or macroeconomic crises, which highly fluctuate financial markets.

---

## Tech Stack & Tools
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Statsmodels, Seaborn, Matplotlib, SciPy
* **Statistical Techniques:** Stepwise Regression, VIF, Durbin-Watson Test, Shapiro-Wilk Test, StandardScaler

# โปรเจกต์วิเคราะห์และคาดการณ์ราคาทองคำในประเทศไทย (Gold Price Forecasting in Thailand)

โปรเจกต์นี้เป็นการนำเทคนิค Data Science มาใช้ในการวิเคราะห์ปัจจัยทางเศรษฐกิจที่มีผลกระทบต่อราคาทองคำในประเทศไทย (ข้อมูลรายเดือน ตั้งแต่ปี 2014 - 2023 รวม 120 ชุด) เพื่อสร้างโมเดลพยากรณ์ราคาทองคำ

# โครงสร้างข้อมูลและตัวแปรที่ใช้ศึกษา
* **ตัวแปรตาม (Target):** ราคาทองคำในประเทศไทย (GPT(THB))
* **ตัวแปรอิสระ (Features):**
  * ราคาทองคำโลก (GPW(USD))
  * ดัชนีราคาผู้บริโภค (CPI)
  * อัตราดอกเบี้ยนโยบาย (IR)
  * ราคาน้ำมันดิบโลก (POIL(USD))
  * อัตราแลกเปลี่ยนเงินบาทต่อดอลลาร์สหรัฐ (EXR)

---

## ขั้นตอนและกระบวนการทำโปรเจกต์ (Methodology)

### 1. การเตรียมข้อมูล (Data Preprocessing)
* ตรวจสอบชนิดข้อมูลและค่าสูญหาย (Missing Values) พบว่าข้อมูลมีความสมบูรณ์ 100% ไม่มีค่าว่าง
* ทำการแบ่งชุดข้อมูลเป็น Train Set (70%) และ Test Set (30%) เพื่อป้องกันปัญหา Overfitting

### 2. การคัดเลือกตัวแปร (Feature Selection)
* ใช้เทคนิค **Stepwise Regression** ในการคัดเลือกตัวแปรอิสระที่มีนัยสำคัญทางสถิติสูงที่สุดต่อราคาทองคำไทย ซึ่งผลลัพธ์คัดเลือกออกมาได้ 3 ตัวแปรหลัก ได้แก่:
  1. ราคาทองคำโลก (GPW(USD))
  2. อัตราแลกเปลี่ยน (EXR)
  3. ดัชนีราคาผู้บริโภค (CPI)

### 3. การทดสอบสมมติฐานทางสถิติ (Assumptions Checking)
ก่อนจะนำโมเดล Linear Regression ไปใช้งาน ได้มีการตรวจสอบความถูกต้องตามหลักสถิติอย่างเข้มงวด:
* **Multicollinearity (VIF):** ตัวแปรอิสระทั้ง 3 ตัวมีค่า VIF อยู่ระหว่าง 1.64 - 3.77 ซึ่ง**ต่ำกว่า 5 ทั้งหมด** แสดงว่าไม่มีปัญหาตัวแปรแย่งกันอธิบาย (No Multicollinearity)
* **Autocorrelation (Durbin-Watson):** ค่า Durbin-Watson ที่ได้จากโมเดลมีค่า **2.186** (และในขั้นทดสอบได้ 2.026) ซึ่งวิ่งอยู่รอบค่า 2.0 สรุปได้ว่า**ไม่มีปัญหาความสัมพันธ์กันเองของความคลาดเคลื่อน (No Autocorrelation)**
* **Normality of Residuals:** ตรวจสอบความถูกต้องของการกระจายตัวด้วย Shapiro-Wilk Test ได้ค่า p-value เท่ากับ 0.537 (มากกว่า 0.05) แสดงว่าข้อมูลเศษเหลือมีการกระจายตัวเป็นเส้นตรงแบบปกติ (Normal Distribution)

---

## ผลการทดลองและโมเดลที่ดีที่สุด (Model Evaluation)

โปรเจกต์นี้ได้ทำการทดลองเปรียบเทียบประสิทธิภาพการพยากรณ์ผ่าน 3 กลุ่มโมเดลหลัก ได้แก่ Linear Regression, Non-Linear Regression (XGBoost, Decision Tree, Random Forest) และ Time Series Forecasting (ARIMA) โดยวัดผลด้วยค่า **Mean Squared Error (MSE)**

### ตารางเปรียบเทียบประสิทธิภาพโมเดล
| ลำดับ | โมเดล (Model) | ค่า MSE (Mean Squared Error) |
| :---: | :--- | :---: |
| **1** | **Linear Regression (Best Model)** | **98,671.87** |
| 2 | XGBoost Regressor | 151,910.19 |
| 3 | Random Forest Regressor | 342,988.98 |
| 4 | Decision Tree Regressor | 739,270.83 |
| 5 | ARIMA (Time Series) | 49,236,132.33 |

### สรุปผลการวิจัย (Conclusion)
* **โมเดลที่ดีที่สุด:** **Linear Regression** เป็นโมเดลที่ให้ค่าความคลาดเคลื่อน (MSE) ต่ำที่สุดอย่างเห็นได้ชัด และให้ค่า **R-squared สูงถึง 0.994** (อธิบายความผันผวนของราคาทองคำไทยได้ถึง 99.4%)
* **บทวิเคราะห์เพิ่มเติม:** เนื่องจากการทำ Stepwise Regression และการทำ Standardization (StandardScaler) ช่วยให้ตัวแปรอิสระทั้ง 3 ตัวมีความสอดคล้องและถูกต้องตามสมมติฐานทางสถิติทั้งหมด โมเดลเส้นตรงจึงสามารถคำนวณความสัมพันธ์เชิงเส้นระหว่างราคาทองคำโลกและอัตราแลกเปลี่ยนที่มีผลโดยตรงต่อราคาทองคำไทยได้อย่างแม่นยำมากกว่าโมเดลกลุ่ม Tree-based ที่มักจะ Overfit ได้ง่ายในชุดข้อมูลลักษณะนี้ และเนื่องจากโปรเจคในครั้งนี้ไม่ได้มีการนำตัวแปรที่ส่งผลกระทบต่อราคาทองคำบางตัวมาใช้ อาทิเช่น ตัวแปรที่สอดข้องกับภาวะสงคราม โรคระบาด และการเกิดวิกฤตการณ์ทางเศรษฐกิจ จึงส่งผลให้ MSE มีความคลาดเคลื่อนสูง 

---

## ทักษะและเครื่องมือที่ใช้ (Skills & Tools)
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Statsmodels, Seaborn, Matplotlib, SciPy
* **Statistical Techniques:** Stepwise Regression, VIF, Durbin-Watson Test, Shapiro-Wilk Test, StandardScaler
