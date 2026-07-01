#!/usr/bin/env bash
set -u

export DEBIAN_FRONTEND=noninteractive

log() {
  echo "[install-extra-tools] $*"
}

run_or_warn() {
  local desc="$1"
  shift
  log "START: ${desc}"
  if "$@"; then
    log "OK: ${desc}"
  else
    log "WARN: ${desc} failed"
    return 1
  fi
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

ensure_apt_packages() {
  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends "$@"
}

install_node_if_needed() {
  if have_cmd node && have_cmd npm; then
    log "Node.js and npm already installed"
    return 0
  fi

  log "Installing Node.js and npm"
  ensure_apt_packages ca-certificates curl gnupg

  if ! have_cmd node || ! have_cmd npm; then
    # Fallback to distro packages for broad compatibility in containers.
    ensure_apt_packages nodejs npm
  fi

  have_cmd node && have_cmd npm
}

install_gh_if_needed() {
  if have_cmd gh; then
    log "GitHub CLI already installed"
    return 0
  fi

  log "Installing GitHub CLI"
  ensure_apt_packages gh
  have_cmd gh
}

install_global_npm_pkg() {
  local package_name="$1"
  local binary_name="$2"

  if have_cmd "$binary_name"; then
    log "${binary_name} already installed"
    return 0
  fi

  log "Installing npm package ${package_name}"
  sudo npm install -g "$package_name"
  have_cmd "$binary_name"
}

install_opencode() {
  if have_cmd opencode; then
    log "opencode already installed"
    return 0
  fi

  # Try known package names; keep going if one is unavailable.
  local candidates=("@opencode-ai/cli" "opencode-ai" "opencode")
  local pkg
  for pkg in "${candidates[@]}"; do
    if npm view "$pkg" version >/dev/null 2>&1; then
      if sudo npm install -g "$pkg"; then
        if have_cmd opencode; then
          log "Installed opencode from ${pkg}"
          return 0
        fi
      fi
    fi
  done

  log "WARN: Could not install opencode CLI from known npm packages"
  return 1
}

install_antigravity_cli() {
  if have_cmd agy; then
    log "agy already installed"
  else
    log "Installing Antigravity CLI (agy)"
    # Official installer for macOS/Linux.
    if ! curl -fsSL https://antigravity.google/cli/install.sh | bash; then
      log "WARN: Failed to install Antigravity CLI via official installer"
      return 1
    fi
  fi

  # Ensure agy is available from standard PATH in non-login shells.
  if [ -x "$HOME/.local/bin/agy" ] && [ ! -x /usr/local/bin/agy ]; then
    sudo ln -sf "$HOME/.local/bin/agy" /usr/local/bin/agy
  fi

  if ! have_cmd agy; then
    log "WARN: agy binary not found after installation"
    return 1
  fi

  return 0
}

install_copilot_cli() {
  if have_cmd copilot && copilot --version >/dev/null 2>&1; then
    log "copilot already installed"
    return 0
  fi

  if ! have_cmd npm; then
    log "WARN: npm is required for Copilot CLI installation"
    return 1
  fi

  # Official npm package from GitHub docs.
  if ! sudo npm_config_ignore_scripts=false npm install -g @github/copilot --no-audit --no-fund; then
    log "WARN: Failed to install @github/copilot"
    return 1
  fi

  if have_cmd copilot; then
    return 0
  fi

  log "WARN: copilot binary not found after npm install"
  return 1
}

install_gcloud_if_needed() {
  if have_cmd gcloud; then
    log "gcloud already installed"
    return 0
  fi

  log "Installing Google Cloud CLI"
  ensure_apt_packages ca-certificates curl gnupg apt-transport-https

  if [ ! -f /usr/share/keyrings/cloud.google.gpg ]; then
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  fi

  if [ ! -f /etc/apt/sources.list.d/google-cloud-sdk.list ]; then
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
      | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null
  fi

  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends google-cloud-cli

  have_cmd gcloud
}

main() {
  local failures=0

  run_or_warn "Node.js/npm available" install_node_if_needed || failures=$((failures + 1))
  run_or_warn "GitHub CLI available" install_gh_if_needed || failures=$((failures + 1))

  run_or_warn "Claude Code CLI installed" install_global_npm_pkg "@anthropic-ai/claude-code" "claude" || failures=$((failures + 1))
  run_or_warn "Codex CLI installed" install_global_npm_pkg "@openai/codex" "codex" || failures=$((failures + 1))
  run_or_warn "OpenCode CLI installed" install_opencode || failures=$((failures + 1))
  run_or_warn "Antigravity CLI installed" install_antigravity_cli || failures=$((failures + 1))
  run_or_warn "Copilot CLI installed" install_copilot_cli || failures=$((failures + 1))
  run_or_warn "Google Cloud CLI installed" install_gcloud_if_needed || failures=$((failures + 1))

  if [ "$failures" -gt 0 ]; then
    log "Completed with ${failures} warning(s)."
    return 0
  fi

  log "All requested tools are installed."
  return 0
}

main "$@"
