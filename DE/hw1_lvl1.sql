1) 
SELECT *
FROM users

SELECT name, date
FROM users

SELECT id
FROM users

SELECT name AS first_name, date
FROM users

2) 
SELECT DISTINCT *
FROM user_transactions

SELECT DISTINCT user_id
FROM user_transactions


SELECT DISTINCT product_id
FROM user_transactions

3)
SELECT COUNT(*) AS total_employees
FROM employees

SELECT COUNT(name) AS total_employees
FROM employees

4) 
SELECT *
FROM user_transactions
WHERE user_id = 5 OR user_id = 4

SELECT amount
FROM user_transactions
WHERE user_id = 5 OR user_id = 6

SELECT *
FROM user_transactions
WHERE amount > 200 OR product_id = 1

SELECT *
FROM user_transactions
WHERE amount < 500 OR user_id = 5

SELECT *
FROM user_transactions
WHERE amount >= 500 OR user_id > 2

SELECT *
FROM user_transactions
WHERE date = '2005-12-31' OR date = '2020-03-17'

SELECT *
FROM user_transactions
WHERE date > '2003-12-31' OR id > 1

SELECT *
FROM user_transactions
WHERE date < '2005-12-31' OR date > '2001-01-01'

SELECT *
FROM user_transactions
WHERE date < '2005-12-31' OR amount > 600

SELECT *
FROM user_transactions
WHERE product_id = 1 OR product_id = 2 OR date > '2020-01-01'

5)
SELECT * 
FROM user_transactions
WHERE user_id = 5

SELECT amount 
FROM user_transactions
WHERE user_id = 5

SELECT amount 
FROM user_transactions
WHERE product_id = 2

SELECT * 
FROM user_transactions
WHERE amount > 200

SELECT * 
FROM user_transactions
WHERE amount < 500

SELECT * 
FROM user_transactions
WHERE amount >= 500

SELECT * 
FROM user_transactions
WHERE date = '2005-12-31'

SELECT * 
FROM user_transactions
WHERE date > '2003-12-31'

SELECT * 
FROM user_transactions
WHERE date < '2005-12-31'

6)
SELECT MIN(amount) 
FROM user_transactions

SELECT MAX(amount)
FROM user_transactions

SELECT AVG(amount)
FROM user_transactions

SELECT ROUND(AVG(amount))
FROM user_transactions

SELECT ROUND(SUM(amount), 1)
FROM user_transactions

SELECT CEIL(AVG(amount)) 
FROM user_transactions

SELECT FLOOR(AVG(amount)) 
FROM user_transactions

SELECT ROUND(SUM(amount), 4) 
FROM user_transactions

7)
SELECT ABS(SUM(temperature))
FROM weather_data

SELECT *,SIGN(temperature) AS temperature_sign
FROM weather_data

8)
SELECT id, number1 % 2 
FROM numbers

SELECT id, number2 % 3 
FROM numbers

SELECT id, number3 % number1 
FROM numbers

SELECT id, ROUND(SQRT(number3)) 
FROM numbers

SELECT id, POW(number3, number1) 
FROM numbers

SELECT id,
       RANDOM(number1) 
       RANDOM(number2)
       RANDOM(number3) 
       RANDOM(number4) 
FROM numbers

9)
SELECT GREATEST(
  (SELECT MAX(number1) FROM numbers),
  (SELECT MAX(number2) FROM numbers),
  (SELECT MAX(number3) FROM numbers)
) AS max_value
FROM numbers

SELECT LEAST(
  (SELECT MIN(number1) FROM numbers),
  (SELECT MIN(number2) FROM numbers),
  (SELECT MIN(number3) FROM numbers)
) AS min_value
FROM numbers

10)
INSERT INTO users (name, date)
VALUES ('user2', '1999-12-12')

INSERT INTO users (name, date)
VALUES ('user3', '1970-10-06')







