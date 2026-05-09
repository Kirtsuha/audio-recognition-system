# Audio Recognition System

Веб-система для распознавания музыки по короткому аудиофрагменту. Пользователь может загрузить `.mp3`/`.wav` файл или записать звук с микрофона в браузере, а система найдет наиболее вероятный трек в локальном каталоге и сохранит результат в истории.

## Возможности

- регистрация и вход пользователей;
- распознавание аудио по файлу и записи с микрофона;
- поиск треков по названию и исполнителю;
- прослушивание найденного эталонного аудио;
- история распознаваний пользователя;
- административная загрузка одного трека;
- массовый импорт ZIP-архива с CSV-manifest;
- запуск синхронизации каталога и полной перестройки артефактов;
- расчет метрик качества через evaluation-сервис.

## Состав проекта

- `frontend` - React/Vite веб-интерфейс.
- `orchestration-service` - Java/Spring Boot API, авторизация, история, пользовательские и административные сценарии.
- `fingerprint-service` - FastAPI-сервис аудиофингерпринтинга, индексирования и первичного поиска кандидатов.
- `ml-reranker-service` - FastAPI-сервис повторного ранжирования fingerprint-кандидатов.
- `ml-pipeline-runner` - сервис построения и обновления ML-артефактов каталога.
- `evaluation-service` - сервис подготовки проверочных запросов и расчета метрик.
- `PostgreSQL` - базы данных пользовательского контура и fingerprint-каталога.
- `MinIO` - S3-совместимое хранилище аудио, запросов и артефактов.

## Запуск

Из корня репозитория:

```powershell
docker compose up --build -d
```

Проверка состояния:

```powershell
docker compose ps
```

Основные адреса:

- Web UI: [http://localhost:3000](http://localhost:3000)
- Orchestration API: [http://localhost:8080](http://localhost:8080)
- Swagger UI: [http://localhost:8080/swagger-ui/index.html](http://localhost:8080/swagger-ui/index.html)
- Fingerprint service: [http://localhost:8000](http://localhost:8000)
- ML pipeline runner: [http://localhost:8011](http://localhost:8011)
- Evaluation service: [http://localhost:8020](http://localhost:8020)
- ML reranker service: [http://localhost:8030](http://localhost:8030)
- MinIO Console: [http://localhost:9001](http://localhost:9001), login `minio`, password `minio123`

## Типовой сценарий

1. Открыть [http://localhost:3000](http://localhost:3000).
2. Зарегистрироваться или войти в существующую учетную запись.
3. Открыть раздел `Recognition`.
4. Загрузить аудиофайл или записать фрагмент с микрофона.
5. Отправить аудио на распознавание.
6. Посмотреть найденный трек, confidence, source и запись в истории.

## Администрирование каталога

Пользователь с ролью `ROLE_ADMIN` видит раздел `Admin`. В нем можно загрузить один трек, импортировать ZIP-архив с CSV-manifest, запустить `Incremental sync`, выполнить `Full rebuild` и проверить статус задания по `Job ID`.

Manifest для массового импорта должен содержать как минимум колонки:

```csv
path,title,artist
tracks/example.wav,Example track,Example artist
```

После загрузки новых треков нужно запустить синхронизацию каталога, чтобы обновились fingerprint-индекс и артефакты reranker.

## Полезные команды

Логи основных сервисов:

```powershell
docker compose logs -f orchestration-service
docker compose logs -f fingerprint-service
docker compose logs -f ml-reranker-service
docker compose logs -f frontend
```

Остановка стенда:

```powershell
docker compose down
```

Полная остановка с удалением данных:

```powershell
docker compose down -v
```

Команда `docker compose down -v` удаляет базы данных, MinIO-хранилище и построенные артефакты, поэтому использовать ее стоит только для полного сброса стенда.
