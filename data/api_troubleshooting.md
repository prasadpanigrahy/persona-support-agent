# Developer API Support Documentation
[cite_start]All incoming developer requests must include a valid Bearer Token in the HTTP Authorization header[cite: 144].

### Header Layout Specifications:
- `Authorization`: `Bearer <YOUR_SECRET_KEY>`
- `Content-Type`: `application/json`

### Standard Server Status Codes:
- [cite_start]**401 Unauthorized Error**: Triggered when the security bearer token is missing, expired, or malformed.
- **403 Forbidden**: Token lacks necessary administrative read/write scopes.
- [cite_start]**500 Internal Error**: Bad formatting parameters sent to the database backend[cite: 144].