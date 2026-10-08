# Участнику

1. Прочитайте [`../docs/task.md`](../docs/task.md) и
   [`../docs/ontology.md`](../docs/ontology.md) — там задание и точные
   определения классов.
2. Возьмите за основу либо `solution.py` из этой директории (минимальный
   скелет), либо `../baseline/` (рабочий baseline, score 0.712).
3. Проверьте формат ответа локально:

```bash
python solution.py --train ../data/train.csv --test ../data/test.csv \
       --output predictions.csv
python ../scripts/validate_submission.py --pred predictions.csv --test ../data/test.csv
```

4. Оцените себя честно, без ground truth теста:

```bash
python ../scripts/local_cv.py --train ../data/train.csv --folds 3
```

5. Убедитесь, что решение работает из чистой директории: скопируйте свои
   `.py`-файлы вместе с `train.csv` и `test.csv` в пустую папку и запустите
   контрактную команду там — ровно так его запустит платформа.

Требования к сабмиту: один файл `solution.py` (плюс, при необходимости,
модули рядом с ним), аргументы `--train/--test/--output`, без интернета,
≤ 30 минут на всё.
