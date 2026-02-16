CREATE DATABASE fingerprint_db;
CREATE DATABASE orchestration_db;

CREATE USER fingerprint_user WITH PASSWORD 'fingerprint_pass';
CREATE USER orchestration_user WITH PASSWORD 'orchestration_pass';

GRANT ALL PRIVILEGES ON DATABASE fingerprint_db TO fingerprint_user;
GRANT ALL PRIVILEGES ON DATABASE orchestration_db TO orchestration_user;
