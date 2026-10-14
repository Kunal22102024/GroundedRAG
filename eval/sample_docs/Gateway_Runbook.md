---
title: Gateway Runbook
date: 2026-02-10
---

# Gateway Errors

When the gateway returns ERR_404_NET the upstream service could not be reached; check the load balancer first.

## Retry limits

The API retry limit is 3 attempts with exponential backoff starting at 200 ms.

- First retry after 200 ms
- Second retry after 400 ms

```
retry --max 3 --backoff 200ms
```

## Reboot window

Emergency reboots are allowed between 02:00 and 04:00 UTC on Sundays only.
