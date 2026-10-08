# Coordination Referee — стартовый комплект участника

Соревнование по диагностике сбоёв координации в мультиагентных AI-системах.

Мультиагентные системы выполняют бизнес-процессы целиком: закупки, поддержку,
продажи, юридическую проверку. Когда такой запуск проваливается, отдельные
агенты обычно отрабатывают корректно — ломается координация между ними: задачу
передали и потеряли, двое сделали одну работу, команда ушла от цели, процесс
зациклился.

По логу одного запуска нужно предсказать три вещи: класс сбоя координации
(7 классов), достигнута ли бизнес-цель и на каком ходу сбой начался.

## Быстрый старт

```bash
pip install -r requirements.txt
bash scripts/prepare_data.sh

python baseline/solution.py --train data/train.csv --test data/test.csv \
       --output predictions.csv
python scripts/validate_submission.py --pred predictions.csv --test data/test.csv
```

Локальная оценка без разметки теста — кросс-валидация по train:

```bash
python scripts/local_cv.py --train data/train.csv --folds 3
```

## Документация

| Документ | О чём |
|---|---|
| [`docs/task.md`](docs/task.md) | задание: формат ответа, ограничения, примеры сигнала, идеи для улучшений |
| [`docs/ontology.md`](docs/ontology.md) | определения 7 классов и границы между ними — читать обязательно |
| [`docs/data_schema.md`](docs/data_schema.md) | поля, JSON-схемы, шум, сдвиг распределения |
| [`docs/metrics.md`](docs/metrics.md) | формула итогового скора |
| [`docs/baseline.md`](docs/baseline.md) | как устроен baseline и где он слаб |
| [`docs/faq.md`](docs/faq.md) | частые вопросы: веса моделей, предобучение, смоук-прогон |

## Что внутри

```
data/         демо-датасет для локальной разработки (в git — сжатый)
baseline/     рабочий baseline: 122 признака + LightGBM, score 0.712
participant/  минимальный шаблон solution.py
scripts/      распаковка данных, проверка формата, кросс-валидация
evaluation/   определения метрик для локальной валидации
docs/         документация
tests/        проверки стартового комплекта
```

Боевые `train.csv` и `test.csv` выдаются платформой в разделе «Данные»;
датасет в `data/` — демонстрационный, той же природы.

## Окружение запуска

Решение работает в подготовленном образе, **интернета внутри job'а нет**.
Установлены numpy, pandas, scipy, scikit-learn, LightGBM; в GPU-варианте
дополнительно torch 2.5.1 + CUDA 12.4, transformers, sentence-transformers,
xgboost, catboost.

Предобученных моделей в образе нет. Нужна готовая модель — кладите её веса
в архив сабмита рядом с `solution.py` и загружайте по относительному пути.

## Тесты

```bash
pip install -r requirements-dev.txt
pytest tests -q
```
