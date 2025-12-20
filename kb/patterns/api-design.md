---
title: API Design Guidelines
tags:
  - patterns
  - api
  - rest
  - design
created: 2024-12-19
---

# API Design Guidelines

Best practices for designing REST APIs at voidlabs.

## URL Structure

- Use nouns, not verbs: `/users` not `/getUsers`
- Use plural: `/users/{id}` not `/user/{id}`
- Nest for relationships: `/users/{id}/orders`

## HTTP Methods

| Method | Purpose | Idempotent |
|--------|---------|------------|
| GET | Read | Yes |
| POST | Create | No |
| PUT | Replace | Yes |
| PATCH | Update | No |
| DELETE | Remove | Yes |

## Response Codes

- 200 OK - Success
- 201 Created - Resource created
- 400 Bad Request - Client error
- 404 Not Found - Resource doesn't exist
- 500 Internal Server Error - Server error

## Pagination

Use limit/offset or cursor-based pagination for lists.

```json
{
  "data": [...],
  "pagination": {
    "total": 100,
    "limit": 20,
    "offset": 0,
    "next": "/users?limit=20&offset=20"
  }
}
```

## Related

- [[architecture/microservices-patterns]]
