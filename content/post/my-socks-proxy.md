---
title: "Building a SOCKS Proxy with Shadowsocks"
date: 2025-12-20T21:49:45+08:00

---

# Building a SOCKS Proxy with Shadowsocks

This guide walks you through setting up a Shadowsocks proxy server on an ECS instance.

## Step 1: Set Up Your ECS Instance

First, purchase an ECS instance running CentOS. I used Alibaba Cloud, which offers 200GB/month of free public network traffic with CDT.

## Step 2: Install and Configure Shadowsocks

Log into your ECS instance as root and run the following commands:

```bash
yum install -y docker
docker run -d \
  --name ss-server \
  -p 8388:8388 \
  -e PASSWORD=${use-your-password} \
  -e METHOD=aes-256-cfb \
  shadowsocks/shadowsocks-libev
```

**Note:** Replace `${use-your-password}` with your desired password.

## Step 3: Connect from Your Client

Finally, use ShadowsocksX-NG-R8 (or any compatible Shadowsocks client) to connect from your laptop or desktop computer.