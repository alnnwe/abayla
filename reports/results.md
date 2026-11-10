# Результаты на test (сгенерировано abayla.evaluation.evaluate)

**Демонстрационные синтетические данные.** Эти числа проверяют, что конвейер работает и модели сравниваются честно. Они не являются оценкой качества на реальных сообщениях.

Test: n=40, scam=19, групп=20, языки={'ru': 20, 'kk': 19, 'mixed': 1}. Оборудование: Linux 6.18.44-fc-v338, x86_64, 1 CPU, RAM 3.9 GB, нет GPU, Python 3.12.3.

| Модель | Порог (val) | Precision | Recall | F1 | PR-AUC | ROC-AUC | 95% ДИ recall | 95% ДИ precision | Цели P≥0.85 и R≥0.90 |
|---|---|---|---|---|---|---|---|---|---|
| rules | 0.266 | 0.889 | 0.842 | 0.865 | 0.921 | 0.921 | [0.60; 1.00] | [0.64; 1.00] | нет |
| tfidf_word | 0.590 | 0.739 | 0.895 | 0.810 | 0.698 | 0.810 | [0.75; 1.00] | [0.48; 0.95] | нет |
| tfidf_wordchar | 0.539 | 0.680 | 0.895 | 0.773 | 0.933 | 0.922 | [0.75; 1.00] | [0.41; 0.92] | нет |
| xlmr | 0.036 | 0.720 | 0.947 | 0.818 | 0.945 | 0.940 | [0.83; 1.00] | [0.46; 0.93] | нет |
| models_aug/tfidf_wordchar | 0.619 | 0.739 | 0.895 | 0.810 | 0.921 | 0.912 | [0.75; 1.00] | [0.48; 0.95] | нет |

ROC-AUC классификатора «только длина текста»: 0.788 (проверка «короткого пути»).

## Матрицы ошибок (TN FP / FN TP)

- rules: TN=19 FP=2 / FN=3 TP=16
- tfidf_word: TN=15 FP=6 / FN=2 TP=17
- tfidf_wordchar: TN=13 FP=8 / FN=2 TP=17
- xlmr: TN=14 FP=7 / FN=1 TP=18
- models_aug/tfidf_wordchar: TN=15 FP=6 / FN=2 TP=17

## Три уровня риска (по порогам low/high с validation)

- rules: scam → {'low': 3, 'uncertain': 0, 'high': 16}; not_scam → {'low': 19, 'uncertain': 0, 'high': 2}
- tfidf_word: scam → {'low': 2, 'uncertain': 16, 'high': 1}; not_scam → {'low': 12, 'uncertain': 8, 'high': 1}
- tfidf_wordchar: scam → {'low': 2, 'uncertain': 0, 'high': 17}; not_scam → {'low': 13, 'uncertain': 4, 'high': 4}
- xlmr: scam → {'low': 1, 'uncertain': 3, 'high': 15}; not_scam → {'low': 9, 'uncertain': 10, 'high': 2}
- models_aug/tfidf_wordchar: scam → {'low': 2, 'uncertain': 0, 'high': 17}; not_scam → {'low': 13, 'uncertain': 4, 'high': 4}

## По языкам (размеры подвыборок указаны)

- rules: kk: n=19 (scam 9), recall=0.889, FPR=0.1; mixed: n=1 (scam 1), recall=1.0, FPR=—; ru: n=20 (scam 9), recall=0.778, FPR=0.091
- tfidf_word: kk: n=19 (scam 9), recall=0.778, FPR=0.3; mixed: n=1 (scam 1), recall=1.0, FPR=—; ru: n=20 (scam 9), recall=1.0, FPR=0.273
- tfidf_wordchar: kk: n=19 (scam 9), recall=0.778, FPR=0.4; mixed: n=1 (scam 1), recall=1.0, FPR=—; ru: n=20 (scam 9), recall=1.0, FPR=0.364
- xlmr: kk: n=19 (scam 9), recall=0.889, FPR=0.3; mixed: n=1 (scam 1), recall=1.0, FPR=—; ru: n=20 (scam 9), recall=1.0, FPR=0.364
- models_aug/tfidf_wordchar: kk: n=19 (scam 9), recall=0.778, FPR=0.2; mixed: n=1 (scam 1), recall=1.0, FPR=—; ru: n=20 (scam 9), recall=1.0, FPR=0.364

## Recall по схемам

- rules: bank_police_impersonation: 1.0 (n=4); fake_ad: 1.0 (n=2); investment: 0.0 (n=2); other_scheme: 0.75 (n=4); phishing: 1.0 (n=4); relative_in_trouble: 1.0 (n=3)
- tfidf_word: bank_police_impersonation: 1.0 (n=4); fake_ad: 1.0 (n=2); investment: 0.5 (n=2); other_scheme: 1.0 (n=4); phishing: 1.0 (n=4); relative_in_trouble: 0.667 (n=3)
- tfidf_wordchar: bank_police_impersonation: 1.0 (n=4); fake_ad: 1.0 (n=2); investment: 0.5 (n=2); other_scheme: 1.0 (n=4); phishing: 1.0 (n=4); relative_in_trouble: 0.667 (n=3)
- xlmr: bank_police_impersonation: 1.0 (n=4); fake_ad: 1.0 (n=2); investment: 1.0 (n=2); other_scheme: 1.0 (n=4); phishing: 1.0 (n=4); relative_in_trouble: 0.667 (n=3)
- models_aug/tfidf_wordchar: bank_police_impersonation: 1.0 (n=4); fake_ad: 1.0 (n=2); investment: 0.5 (n=2); other_scheme: 1.0 (n=4); phishing: 1.0 (n=4); relative_in_trouble: 0.667 (n=3)

## Задержка (Analyzer.analyze: модель + признаки + URL; без сети Telegram и без LLM)

| Модель | Вызовов | Медиана, мс | p95, мс | Загрузка, с | p95 < 5 с |
|---|---|---|---|---|---|
| rules | 200 | 0.48 | 0.68 | 0.0 | да |
| tfidf_word | 200 | 1.3 | 1.51 | 0.012 | да |
| tfidf_wordchar | 200 | 2.86 | 3.29 | 0.051 | да |
| xlmr | 40 | 104.22 | 133.76 | 3.137 | да |
| models_aug/tfidf_wordchar | 200 | 2.98 | 3.44 | 0.039 | да |

## Уровни в сервисе (с политикой «сильный признак не даёт низкий риск»)

- только модель: {'not_scam': {'low': 13, 'uncertain': 4, 'high': 4}, 'scam': {'low': 2, 'uncertain': 0, 'high': 17}}
- с политикой: {'not_scam': {'low': 13, 'uncertain': 4, 'high': 4}, 'scam': {'low': 2, 'uncertain': 0, 'high': 17}}

## Learning curve (на validation)

- 19 групп: PR-AUC 0.833 ± 0.066, F1@0.5 0.73
- 38 групп: PR-AUC 0.878 ± 0.034, F1@0.5 0.837
- 57 групп: PR-AUC 0.936 ± 0.037, F1@0.5 0.855
- 76 групп: PR-AUC 0.961 ± 0.016, F1@0.5 0.865
- 95 групп: PR-AUC 0.978 ± 0.0, F1@0.5 0.868

Синтетика vs реальные данные: не выполнено: все примеры synthetic_demo, реальных сообщений в наборе нет.

Ошибочные предсказания для ручного анализа: reports/errors/*.csv. Графики: reports/figures/.
