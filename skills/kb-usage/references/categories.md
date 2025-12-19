# KB Category Taxonomy

The knowledge base is organized into top-level categories. Each entry belongs to exactly one category based on its primary purpose.

## Categories

### infrastructure/

Physical and cloud infrastructure, networking, and core services.

**Use for:**
- Cloud provider configurations (AWS, GCP, Azure)
- DNS setup and management
- Networking (VPCs, subnets, firewalls, load balancers)
- Server provisioning and management
- Storage systems and configuration
- SSL/TLS certificates
- Domain management

**Example entries:**
- `cloudflare-dns-setup.md`
- `aws-vpc-architecture.md`
- `ssl-certificate-renewal.md`

**Common tags:** `cloud`, `dns`, `networking`, `servers`, `storage`, `ssl`, `aws`, `gcp`

---

### devops/

CI/CD, deployment, monitoring, and operational tooling.

**Use for:**
- CI/CD pipeline configurations
- Deployment procedures and automation
- Monitoring and alerting setup
- Logging and observability
- Container orchestration operations
- Release management
- Incident response procedures

**Example entries:**
- `github-actions-workflow.md`
- `kubernetes-deployment-strategy.md`
- `prometheus-alerting-rules.md`

**Common tags:** `ci`, `cd`, `monitoring`, `deployment`, `docker`, `kubernetes`, `observability`, `logging`

---

### development/

Languages, frameworks, tooling, and coding conventions.

**Use for:**
- Language-specific conventions (Python, Rust, Go, etc.)
- Framework usage guides
- Development environment setup
- Testing strategies and frameworks
- Code review guidelines
- Tooling configurations (linters, formatters)
- Library and dependency management

**Example entries:**
- `python-project-structure.md`
- `rust-error-handling.md`
- `pre-commit-hooks-setup.md`

**Common tags:** `python`, `rust`, `go`, `typescript`, `testing`, `tooling`, `linting`, `formatting`

---

### troubleshooting/

Known issues, debugging guides, and problem solutions.

**Use for:**
- Documented issues with known solutions
- Debugging procedures
- Error message explanations
- Performance problem diagnosis
- Recovery procedures
- Workarounds and their context

**Example entries:**
- `kubernetes-pod-crashloop.md`
- `dns-propagation-issues.md`
- `memory-leak-debugging.md`

**Common tags:** `debugging`, `errors`, `performance`, `recovery`, `workaround`

---

### architecture/

System design, architectural decisions, and trade-offs.

**Use for:**
- System architecture documentation
- Architectural Decision Records (ADRs)
- Design trade-off analysis
- Integration patterns
- Scalability considerations
- Security architecture

**Example entries:**
- `microservices-communication.md`
- `adr-001-database-choice.md`
- `event-driven-architecture.md`

**Common tags:** `design`, `adr`, `scalability`, `security`, `integration`

---

### patterns/

Reusable code patterns, conventions, and best practices.

**Use for:**
- Code patterns that should be reused
- Design patterns applied to our stack
- Naming conventions
- API design patterns
- Error handling patterns
- Configuration patterns

**Example entries:**
- `repository-pattern-python.md`
- `api-versioning-strategy.md`
- `configuration-management.md`

**Common tags:** `pattern`, `convention`, `api`, `design-pattern`, `best-practice`

---

## Choosing the Right Category

Ask yourself:

1. **What is the primary purpose of this knowledge?**
   - Setting up infrastructure → `infrastructure/`
   - Deploying or monitoring → `devops/`
   - Writing or organizing code → `development/`
   - Fixing a problem → `troubleshooting/`
   - Explaining system design → `architecture/`
   - Documenting reusable patterns → `patterns/`

2. **Who would look for this?**
   - Platform/infra engineer → likely `infrastructure/` or `devops/`
   - Developer → likely `development/` or `patterns/`
   - Anyone debugging → likely `troubleshooting/`
   - Technical lead/architect → likely `architecture/`

3. **When something spans categories:**
   - Pick the PRIMARY purpose
   - Use tags to indicate secondary topics
   - Link to related entries in other categories

## Examples of Category Selection

| Topic | Category | Why |
|-------|----------|-----|
| How to set up Kubernetes cluster | infrastructure/ | Core infrastructure setup |
| How to deploy to Kubernetes | devops/ | Deployment procedure |
| Kubernetes Go client patterns | development/ | Coding conventions |
| Pod keeps crashing | troubleshooting/ | Problem solving |
| Why we chose Kubernetes | architecture/ | Design decision |
| Kubernetes resource patterns | patterns/ | Reusable templates |

## Category Boundaries

**infrastructure/ vs devops/:**
- Infrastructure = the thing itself (cluster, server, network)
- DevOps = operating the thing (deploying, monitoring, maintaining)

**development/ vs patterns/:**
- Development = how we use tools and languages
- Patterns = reusable abstractions and conventions

**troubleshooting/ vs everything else:**
- If it starts with "X is broken" or "how to debug X" → troubleshooting/
- If it explains how X works or how to set up X → appropriate other category
