-- Normalized reference schema for the Rapido datasets, matching the real
-- CSV columns (see PROJECT_IMPLEMENTATION_GUIDE.md section 2 and 7).
--
-- At runtime, src/db.py loads the cleaned data into a local SQLite file via
-- SQLAlchemy's to_sql (which creates its own table DDL). This file is the
-- normalized design reference required by the project guide's SQL
-- practices, and can be run as-is against MySQL/Postgres if the project is
-- migrated off SQLite.

CREATE TABLE locations (
    location_id BIGINT PRIMARY KEY,
    city VARCHAR(50) NOT NULL,
    location_code VARCHAR(20) NOT NULL,
    UNIQUE (city, location_code)
);

CREATE TABLE time_features (
    datetime DATETIME PRIMARY KEY,
    hour_of_day TINYINT NOT NULL,
    day_of_week VARCHAR(10) NOT NULL,
    is_weekend TINYINT NOT NULL,
    is_holiday TINYINT NOT NULL,
    peak_time_flag TINYINT NOT NULL,
    season VARCHAR(20) NOT NULL
);

CREATE TABLE customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    customer_gender VARCHAR(20),
    customer_age TINYINT,
    customer_city VARCHAR(50),
    customer_signup_days_ago INT,
    preferred_vehicle_type VARCHAR(20),
    total_bookings INT,
    completed_rides INT,
    cancelled_rides INT,
    incomplete_rides INT,
    cancellation_rate DECIMAL(6, 5),
    avg_customer_rating DECIMAL(3, 2),
    customer_cancel_flag TINYINT
);

CREATE TABLE drivers (
    driver_id VARCHAR(20) PRIMARY KEY,
    driver_age TINYINT,
    driver_city VARCHAR(50),
    vehicle_type VARCHAR(20),
    driver_experience_years TINYINT,
    total_assigned_rides INT,
    accepted_rides INT,
    incomplete_rides INT,
    delay_count INT,
    acceptance_rate DECIMAL(6, 5),
    delay_rate DECIMAL(6, 5),
    avg_driver_rating DECIMAL(3, 2),
    avg_pickup_delay_min DECIMAL(6, 2),
    driver_delay_flag TINYINT
);

CREATE TABLE bookings (
    booking_id VARCHAR(20) PRIMARY KEY,
    booking_date DATE NOT NULL,
    booking_time TIME NOT NULL,
    day_of_week VARCHAR(10),
    is_weekend TINYINT,
    hour_of_day TINYINT,
    city VARCHAR(50),
    pickup_location_id BIGINT REFERENCES locations(location_id),
    drop_location_id BIGINT REFERENCES locations(location_id),
    vehicle_type VARCHAR(20),
    ride_distance_km DECIMAL(6, 2),
    estimated_ride_time_min DECIMAL(6, 2),
    actual_ride_time_min DECIMAL(6, 2),
    traffic_level VARCHAR(20),
    weather_condition VARCHAR(30),
    base_fare DECIMAL(8, 2),
    surge_multiplier DECIMAL(4, 2),
    booking_value DECIMAL(8, 2),
    booking_status VARCHAR(20),
    incomplete_ride_reason VARCHAR(50),
    customer_id VARCHAR(20) REFERENCES customers(customer_id),
    driver_id VARCHAR(20) REFERENCES drivers(driver_id)
);

CREATE TABLE location_demand (
    city VARCHAR(50),
    pickup_location VARCHAR(20),
    hour_of_day TINYINT,
    vehicle_type VARCHAR(20),
    total_requests INT,
    completed_rides INT,
    cancelled_rides INT,
    avg_wait_time_min DECIMAL(6, 2),
    avg_surge_multiplier DECIMAL(4, 2),
    demand_level VARCHAR(20),
    PRIMARY KEY (city, pickup_location, hour_of_day, vehicle_type)
);

CREATE INDEX idx_bookings_customer_id ON bookings(customer_id);
CREATE INDEX idx_bookings_driver_id ON bookings(driver_id);
CREATE INDEX idx_bookings_booking_date ON bookings(booking_date);
CREATE INDEX idx_bookings_city ON bookings(city);
CREATE INDEX idx_bookings_status ON bookings(booking_status);
