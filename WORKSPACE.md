# Рабочая область Coordination Referee

Стартовый репозиторий: https://github.com/dmayudin/coordination-referee

## Окружение

В `.venv` создано локальное окружение Python 3.12. Команды ниже выполняются
из папки SBER в PowerShell. Активация окружения не требуется.
Установленные версии библиотек зафиксированы в `requirements-local.lock.txt`.
Для восстановления зависимостей в существующем окружении:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-local.lock.txt
```

## Данные

`data/train.csv` и `data/test.csv` распакованы из архивов стартового комплекта.
Это **демонстрационные данные**, а не боевые файлы платформы.
Оригинальные `data/*.csv.gz` сохранены. Боевые файлы скачиваются отдельно
из раздела «Данные» DSWorks; при запуске укажите пути к ним через аргументы.
Предсказания для демо-теста не подходят для отправки на платформу.

Архив пользователя `data/coordination-referee-participant.zip` распакован
в `data/competition/`. Для работы с его данными используйте
`--train data/competition/train.csv --test data/competition/test.csv`.
Train-ID архива совпадают с демо, но обозначают другие запуски: не
объединяйте наборы по одному run_id. Демо-предсказания в корне относятся
к прежнему демонстрационному test.

## Команды

Baseline на полном демо-наборе:

```powershell
.\.venv\Scripts\python.exe baseline/solution.py --train data/train.csv --test data/test.csv --output predictions.csv
.\.venv\Scripts\python.exe scripts/validate_submission.py --pred predictions.csv --test data/test.csv
```

Локальная оценка на обучающей выборке:

```powershell
.\.venv\Scripts\python.exe scripts/local_cv.py --train data/train.csv --folds 3
```

Быстрая пробная оценка (не заменяет оценку на полной выборке):

```powershell
.\.venv\Scripts\python.exe scripts/local_cv.py --train data/train.csv --folds 3 --limit 1000
```

Тесты стартового комплекта:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

## Где работать

Исследовательский план для целевого score ≥ 0.95: [docs/SCORE_095_PLAN.md](docs/SCORE_095_PLAN.md).
Ход реализации и подтверждённые результаты: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

- `baseline/` — готовое решение организаторов: признаки, модели и локализация.
- `referee/` — улучшенные признаки, обучение моделей запусков и ходов.
- `solution.py` — автономная точка входа собственного решения.
- `participant/solution.py` — исходный шаблон для собственного решения.
- `docs/ontology.md` — определения классов и границы между ними.
- `docs/data_schema.md` — структура входных данных.
- `docs/metrics.md` — метрики.
- `docs/faq.md` — уточнения требований; обучение заранее допускается,
  готовых весов моделей в образе нет.

Baseline и шаблон оставлены в исходном виде. `predictions.csv`, окружение,
распакованные данные и временные результаты исключены из Git.
Локальный lock-файл предназначен для Windows; окружение платформы уже готово
и может содержать другие версии библиотек. Проверка формата ответа не измеряет
качество на тесте: его истинная разметка участникам недоступна.

## Проверка подготовки — 7 октября 2026

- Тесты стартового комплекта: 12 passed.
- Полный демо-прогон baseline: 10 000 train, 4 000 test, около 48 секунд.
- `predictions.csv`: 4 000 строк, проверка формата успешна.
- Пробная 3-fold CV на 1 000 строках: score 0.6391. Это результат малой
  демо-выборки, он не является оценкой на лидерборде.
- Полная 3-fold CV на демо-train: локальный score 0.7328.
- Полная 3-fold CV на train из архива пользователя (`data/competition/`):
  локальный score 0.7245. RobustnessF1 в локальном скрипте является
  приближением официального среза.

## Собственное решение

Воспроизвести развитие модели:

```powershell
.\.venv\Scripts\python.exe scripts/experiment.py --name reproduce_v2 --mode struct --profile v2 --turn-model --trees 850 --turn-weight .5 --success-threshold .4 --extra-train data/train.csv
```

Обучить полные веса подтверждённой версии и создать ответ:

```powershell
.\.venv\Scripts\python.exe scripts/fit_solution.py --extra-train data/train.csv --profile v5 --success-threshold .45
.\.venv\Scripts\python.exe solution.py --train data/competition/train.csv --test data/competition/test.csv --output outputs/predictions.csv
.\.venv\Scripts\python.exe scripts/validate_submission.py --pred outputs/predictions.csv --test data/competition/test.csv
```

`outputs/` содержит кеши, локальные модели и метрики. `weights/` содержит
полные веса; обе папки исключены из Git. Текущие веса по умолчанию — v5,
обученные на 20 000 уникальных содержаний official + demo train. Предыдущие
веса v2 сохранены в `weights/v2`, прежний архив `submission-v2.zip` сохранён.

По просьбе пользователя 8 октября зафиксирована версия v5. Лучший локальный
score proxy — **0.914582** на повторном контрольном fold 0 (обучение на
18 000 строках). Другие разбиения дали 0.888858 и 0.884393 при обучении
на 16 000 строках. Повторный контроль уже использовался в разработке;
RobustnessF1 — приближение скрытого среза. Это не результат платформы,
пороги 0.92 и 0.95 не достигнуты.

Архив для загрузки: `submission-v5-fixed.zip` (16.4 MB), CPU-образ,
готовые веса и `solution.py` находятся в корне архива. CSV для просмотра:
`outputs/predictions-v5-fixed.csv`. Отчёт о фиксации и происхождении весов:
`outputs/v5-fixed/validation.json`; проверка архива после отдельной
распаковки: `outputs/v5-fixed/package-validation.json`.
Отдельный запуск занял 171.125 секунды; все 4 000 прогнозов совпали с
рабочей областью, формат и границы fault_turn прошли проверку. 28 тестов
прошли. Точка входа также точно воспроизвела сохранённые прогнозы
контрольных 2 000 строк; отчёт — `outputs/v5-fixed/entry-validation.json`.
Доступ к платформе пока не используется, эксперименты приостановлены.

Дополнительные проверки:

```powershell
.\.venv\Scripts\python.exe scripts/package_submission.py
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp outputs/pytest-verification
.\.venv\Scripts\python.exe scripts/relation_experiment.py --name relations_v5
```

Предобученный энкодер в `weights/encoder` нужен только для эксперимента
`scripts/semantic_experiment.py`. Основное решение его не загружает.
