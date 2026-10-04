# Comprehensive Confusion Matrices — Baseline Ladder

Class Definitions: Low (0), Medium (1), High (2)

### Model A (Horizon B) - Dummy Majority

| Actual \ Predicted | Pred: Low | Pred: Medium | Pred: High | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual: Low** | 3,700 | 0 | 0 | **3,700** |
| **Actual: Medium** | 887 | 0 | 0 | **887** |
| **Actual: High** | 131 | 0 | 0 | **131** |
| **Total Predicted** | **4,718** | **0** | **0** | **4,718** |

### Model A (Horizon B) - Logistic Regression

| Actual \ Predicted | Pred: Low | Pred: Medium | Pred: High | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual: Low** | 961 | 1,155 | 1,584 | **3,700** |
| **Actual: Medium** | 218 | 295 | 374 | **887** |
| **Actual: High** | 27 | 43 | 61 | **131** |
| **Total Predicted** | **1,206** | **1,493** | **2,019** | **4,718** |

### Model A (Horizon B) - Random Forest

| Actual \ Predicted | Pred: Low | Pred: Medium | Pred: High | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual: Low** | 3,027 | 600 | 73 | **3,700** |
| **Actual: Medium** | 721 | 144 | 22 | **887** |
| **Actual: High** | 106 | 21 | 4 | **131** |
| **Total Predicted** | **3,854** | **765** | **99** | **4,718** |

### Model A (Horizon B) - LightGBM

| Actual \ Predicted | Pred: Low | Pred: Medium | Pred: High | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual: Low** | 1,667 | 1,304 | 729 | **3,700** |
| **Actual: Medium** | 386 | 341 | 160 | **887** |
| **Actual: High** | 53 | 51 | 27 | **131** |
| **Total Predicted** | **2,106** | **1,696** | **916** | **4,718** |

### Model A (Horizon A) - Dummy Majority

| Actual \ Predicted | Pred: Low | Pred: Medium | Pred: High | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual: Low** | 15,592 | 0 | 0 | **15,592** |
| **Actual: Medium** | 3,789 | 0 | 0 | **3,789** |
| **Actual: High** | 609 | 0 | 0 | **609** |
| **Total Predicted** | **19,990** | **0** | **0** | **19,990** |

### Model A (Horizon A) - Logistic Regression

| Actual \ Predicted | Pred: Low | Pred: Medium | Pred: High | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual: Low** | 4,010 | 4,868 | 6,714 | **15,592** |
| **Actual: Medium** | 1,031 | 1,139 | 1,619 | **3,789** |
| **Actual: High** | 168 | 171 | 270 | **609** |
| **Total Predicted** | **5,209** | **6,178** | **8,603** | **19,990** |

### Model A (Horizon A) - Random Forest

| Actual \ Predicted | Pred: Low | Pred: Medium | Pred: High | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual: Low** | 9,548 | 4,209 | 1,835 | **15,592** |
| **Actual: Medium** | 2,410 | 947 | 432 | **3,789** |
| **Actual: High** | 376 | 165 | 68 | **609** |
| **Total Predicted** | **12,334** | **5,321** | **2,335** | **19,990** |

### Model A (Horizon A) - LightGBM

| Actual \ Predicted | Pred: Low | Pred: Medium | Pred: High | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual: Low** | 5,957 | 5,515 | 4,120 | **15,592** |
| **Actual: Medium** | 1,513 | 1,324 | 952 | **3,789** |
| **Actual: High** | 228 | 216 | 165 | **609** |
| **Total Predicted** | **7,698** | **7,055** | **5,237** | **19,990** |

### Model B (Diagnosis) - Dummy Majority

| Actual \ Predicted | Pred: Low | Pred: Medium | Pred: High | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual: Low** | 15,599 | 0 | 0 | **15,599** |
| **Actual: Medium** | 3,791 | 0 | 0 | **3,791** |
| **Actual: High** | 610 | 0 | 0 | **610** |
| **Total Predicted** | **20,000** | **0** | **0** | **20,000** |

### Model B (Diagnosis) - Logistic Regression

| Actual \ Predicted | Pred: Low | Pred: Medium | Pred: High | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual: Low** | 4,313 | 4,503 | 6,783 | **15,599** |
| **Actual: Medium** | 1,077 | 1,110 | 1,604 | **3,791** |
| **Actual: High** | 181 | 154 | 275 | **610** |
| **Total Predicted** | **5,571** | **5,767** | **8,662** | **20,000** |

### Model B (Diagnosis) - Random Forest

| Actual \ Predicted | Pred: Low | Pred: Medium | Pred: High | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual: Low** | 9,979 | 4,448 | 1,172 | **15,599** |
| **Actual: Medium** | 2,372 | 1,124 | 295 | **3,791** |
| **Actual: High** | 405 | 169 | 36 | **610** |
| **Total Predicted** | **12,756** | **5,741** | **1,503** | **20,000** |

### Model B (Diagnosis) - LightGBM

| Actual \ Predicted | Pred: Low | Pred: Medium | Pred: High | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual: Low** | 6,107 | 5,410 | 4,082 | **15,599** |
| **Actual: Medium** | 1,483 | 1,314 | 994 | **3,791** |
| **Actual: High** | 250 | 203 | 157 | **610** |
| **Total Predicted** | **7,840** | **6,927** | **5,233** | **20,000** |

### Model A (Unseen Machines) - Dummy Majority

| Actual \ Predicted | Pred: Low | Pred: Medium | Pred: High | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual: Low** | 3,585 | 0 | 0 | **3,585** |
| **Actual: Medium** | 880 | 0 | 0 | **880** |
| **Actual: High** | 128 | 0 | 0 | **128** |
| **Total Predicted** | **4,593** | **0** | **0** | **4,593** |

### Model A (Unseen Machines) - Logistic Regression

| Actual \ Predicted | Pred: Low | Pred: Medium | Pred: High | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual: Low** | 901 | 1,181 | 1,503 | **3,585** |
| **Actual: Medium** | 210 | 280 | 390 | **880** |
| **Actual: High** | 34 | 44 | 50 | **128** |
| **Total Predicted** | **1,145** | **1,505** | **1,943** | **4,593** |

### Model A (Unseen Machines) - Random Forest

| Actual \ Predicted | Pred: Low | Pred: Medium | Pred: High | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual: Low** | 3,120 | 409 | 56 | **3,585** |
| **Actual: Medium** | 761 | 105 | 14 | **880** |
| **Actual: High** | 110 | 17 | 1 | **128** |
| **Total Predicted** | **3,991** | **531** | **71** | **4,593** |

### Model A (Unseen Machines) - LightGBM

| Actual \ Predicted | Pred: Low | Pred: Medium | Pred: High | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual: Low** | 1,734 | 1,230 | 621 | **3,585** |
| **Actual: Medium** | 442 | 286 | 152 | **880** |
| **Actual: High** | 61 | 50 | 17 | **128** |
| **Total Predicted** | **2,237** | **1,566** | **790** | **4,593** |

