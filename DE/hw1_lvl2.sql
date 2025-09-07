+1)
SELECT DISTINCT city
FROM users

SELECT *
FROM users
WHERE salary = (SELECT MAX(salary) FROM users)

SELECT name
FROM users
WHERE salary > (SELECT AVG(salary) FROM users)

SELECT COUNT(*) 
FROM users
WHERE EXTRACT(YEAR FROM registration_date) = EXTRACT(YEAR FROM CURRENT_DATE)

SELECT *
FROM users
WHERE name LIKE 'A%'

2)
SELECT *
FROM users
WHERE name LIKE 'M%a'

SELECT *
FROM users
WHERE CAST(salary AS TEXT) LIKE '%5'

SELECT *
FROM users
WHERE salary < 4000
  AND registration_date = CURRENT_DATE - INTERVAL '3 years'

SELECT *
FROM users
WHERE registration_date = DATE_TRUNC('month', registration_date + INTERVAL '1 month') - INTERVAL '1 day'

SELECT *
FROM users
WHERE name NOT LIKE '% %'

3)
SELECT DISTINCT department
FROM employees
ORDER BY department

SELECT e.first_name, e.last_name, e.salary, e.department
FROM employees e
WHERE e.salary > (
    SELECT AVG(salary)
    FROM employees
    WHERE department = e.department
)
ORDER BY e.department, e.salary DESC

SELECT first_name, last_name, start_work_date
FROM employees
WHERE start_work_date <= CURRENT_DATE - INTERVAL '5 years'
ORDER BY start_work_date

SELECT first_name, last_name
FROM employees
WHERE manager_id IS NULL

SELECT first_name, last_name, start_work_date
FROM employees
ORDER BY start_work_date DESC

4) 
SELECT *
FROM employees
WHERE department IN (
    SELECT department
    FROM employees
    GROUP BY department
    HAVING COUNT(*) < 3
)

SELECT first_name, last_name
FROM employees
WHERE salary::text LIKE '1%'

SELECT *
FROM employees
WHERE manager_id IS NOT NULL AND department = 'PHP'

SELECT *
FROM employees
WHERE salary::text LIKE '%0'

SELECT *
FROM employees e
WHERE start_work_date > CURRENT_DATE - INTERVAL '2 years'
    AND salary > (
     SELECT AVG(salary)
     FROM employees
     WHERE department = e.department
    )
 
5)
SELECT category, COUNT(*) 
FROM products
GROUP BY category

SELECT *
FROM products p
WHERE price = (
    SELECT MAX(price)
    FROM products
    WHERE category = p.category
)

SELECT category, AVG(price) 
FROM products
GROUP BY category

SELECT *
FROM products
WHERE amount < 500

SELECT *
FROM products
WHERE name LIKE 'B%'

7)
SELECT DISTINCT name
FROM orders

SELECT name, COUNT(*) 
FROM orders
GROUP BY name

SELECT *
FROM orders
WHERE EXTRACT(YEAR FROM date) = EXTRACT(YEAR FROM CURRENT_DATE);

SELECT AVG(total) 
FROM orders
WHERE date >= CURRENT_DATE - INTERVAL '3 months'

SELECT *
FROM orders
WHERE total > 1000

6)
SELECT category, COUNT(*) 
FROM products p1
WHERE price < (
    SELECT AVG(price)
    FROM products p2
    WHERE p2.category = p1.category
)
GROUP BY category

SELECT p.*, sub.avg_amount
FROM (
    SELECT category, MIN(price) AS min_price, AVG(amount) AS avg_amount
    FROM products
    GROUP BY category
) sub
JOIN products p ON p.category = sub.category AND p.price = sub.min_price


WITH cat_counts AS (
    SELECT category, COUNT(*) AS cnt
    FROM products
    GROUP BY category
),
filtered AS (
    SELECT p.*
    FROM products p
    JOIN cat_counts c ON p.category = c.category
    WHERE c.cnt > 3
),
medians AS (
    SELECT category,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY price) AS median_price
    FROM filtered
    GROUP BY category
)
SELECT f.category, AVG(f.price) AS avg_price, m.median_price
FROM filtered f
JOIN medians m ON f.category = m.category
GROUP BY f.category, m.median_price;

SELECT *
FROM products
WHERE amount > 500 AND price < 1;

SELECT *
FROM products
WHERE name ~ '^\w+ \w+$' AND price > 1.5

8)
#1
SELECT DISTINCT name, SUM(total), COUNT(*)
FROM orders
GROUP BY name
ORDER BY total_sum ASC

#2
SELECT name, COUNT(*) 
FROM orders
GROUP BY name
HAVING COUNT(*) < 3

#5
SELECT *
FROM orders
WHERE total > 1000

9)
SELECT *
FROM order_details
WHERE product_count > (
    SELECT AVG(product_count)
    FROM order_details
)

SELECT *
FROM order_details
WHERE discount = (
    SELECT MAX(discount)
    FROM order_details
)

SELECT *
FROM order_details
WHERE product_count > 5

10)
SELECT id, name, product_count
FROM order_details
WHERE product_count > 2

SELECT *, (SELECT AVG(discount) FROM order_details) AS avg_discount
FROM order_details
WHERE discount = (SELECT MAX(discount) FROM order_details)

SELECT *
FROM order_details
WHERE discount = (SELECT MAX(discount) FROM order_details)
  AND product_count > 5










