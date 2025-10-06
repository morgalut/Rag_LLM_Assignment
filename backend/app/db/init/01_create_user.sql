DO
$$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'raguser') THEN
        CREATE ROLE raguser WITH LOGIN PASSWORD 'ragpass';
        CREATE DATABASE ragdb OWNER raguser;
    END IF;
END
$$;
