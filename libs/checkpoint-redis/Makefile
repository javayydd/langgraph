.PHONY: format lint test

format:
 poetry run black .
 poetry run isort .

lint:
 poetry run black . --check
 poetry run isort . --check
 poetry run mypy .
 poetry run ruff .

test:
 poetry run pytest

######################
# DOCKER COMMANDS
######################

start-redis:
	docker compose -f tests/compose-redis.yml up -V --force-recreate --wait || ( \
		echo "Failed to start Redis, printing logs..."; \
		docker compose -f tests/compose-redis.yml logs; \
		exit 1 \
	)

stop-redis:
	docker compose -f tests/compose-redis.yml down

test_redis:
	@echo "Testing Redis"
	@make start-redis
	@poetry run pytest $(TEST)
	@EXIT_CODE=$$?; \
	make stop-redis; \
	echo "Finished testing Redis; Exit code: $$EXIT_CODE"; \
	exit $$EXIT_CODE

test_watch_redis:
	make start-redis; \
	poetry run ptw $(TEST); \
	EXIT_CODE=$$?; \
	make stop-redis; \
	exit $$EXIT_CODE