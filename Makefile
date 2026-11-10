.PHONY: install data eda train train-xlmr evaluate test demo bot
install:      ; pip install -r requirements-dev.txt
data:         ; python -m saqbol.data.demo_dataset
eda:          ; python -m saqbol.data.eda --data data/demo/saqbol_demo_v1.csv --out reports/eda
train:        ; python -m saqbol.training.train && python -m saqbol.training.train --models tfidf_wordchar --augment --splits artifacts/splits_aug --out artifacts/models_aug
train-xlmr:   ; python -m saqbol.training.train_transformer --epochs 4
evaluate:     ; python -m saqbol.evaluation.evaluate --extra-models artifacts/models_aug/tfidf_wordchar
test:         ; python -m pytest -q
demo:         ; python -m saqbol.cli scenarios
bot:          ; python -m saqbol.bot.main
