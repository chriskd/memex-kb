---
title: Microservices Architecture Patterns
tags:
  - architecture
  - microservices
  - design-patterns
created: 2024-12-19
---

# Microservices Architecture Patterns

Design patterns for building distributed microservice systems.

## Service Communication

### Synchronous (REST/gRPC)
Direct request-response between services. Simple but creates coupling.

### Asynchronous (Message Queues)
Services communicate via message brokers like RabbitMQ or Kafka. Better decoupling.

## Data Patterns

### Database per Service
Each microservice owns its data. Prevents tight coupling but complicates queries.

### Saga Pattern
Distributed transactions across services using choreography or orchestration.

## Resilience Patterns

- **Circuit Breaker** - Prevent cascade failures
- **Bulkhead** - Isolate failures
- **Retry with backoff** - Handle transient failures

## Related

- [[patterns/api-design]]
