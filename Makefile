.PHONY: install data eda train train-xlmr evaluate test demo bot
install:      ; pip install -r requirements-dev.txt
data:         ; python -m abayla.data.demo_dataset
eda:          ; python -m abayla.data.eda --data data/demo/abayla_demo_v1.csv --out reports/eda
train:        ; python -m abayla.training.train && python -m abayla.training.train --models tfidf_wordchar --augment --splits artifacts/splits_aug --out artifacts/models_aug
train-xlmr:   ; python -m abayla.training.train_transformer --epochs 4
evaluate:     ; python -m abayla.evaluation.evaluate --extra-models artifacts/models_aug/tfidf_wordchar
test:         ; python -m pytest -q
demo:         ; python -m abayla.cli scenarios
bot:          ; python -m abayla.bot.main
