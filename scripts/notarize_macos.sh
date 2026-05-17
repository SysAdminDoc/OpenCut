#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${1:-dist/OpenCut-Server}"
ZIP_PATH="${2:-dist/OpenCut-Server-macOS.zip}"
RUNNER_TEMP_DIR="${RUNNER_TEMP:-/tmp}"
KEYCHAIN_PATH="${RUNNER_TEMP_DIR}/opencut-notary.keychain-db"
KEYCHAIN_PASSWORD="${MACOS_KEYCHAIN_PASSWORD:-opencut-notary-ci}"
CERT_PATH="${RUNNER_TEMP_DIR}/opencut-developer-id.p12"
API_KEY_PATH="${RUNNER_TEMP_DIR}/AuthKey_${APPLE_API_KEY_ID:-missing}.p8"

required_env=(
  MACOS_CERTIFICATE_P12_BASE64
  MACOS_CERTIFICATE_PASSWORD
  APPLE_API_KEY_ID
  APPLE_API_ISSUER_ID
  APPLE_API_PRIVATE_KEY
)

usage() {
  cat <<'EOF'
Usage:
  scripts/notarize_macos.sh [APP_DIR] [ZIP_PATH]
  scripts/notarize_macos.sh --check-env

Required environment:
  MACOS_CERTIFICATE_P12_BASE64  Base64-encoded Developer ID Application .p12
  MACOS_CERTIFICATE_PASSWORD    Password for the .p12
  APPLE_API_KEY_ID              App Store Connect API key id
  APPLE_API_ISSUER_ID           App Store Connect issuer id
  APPLE_API_PRIVATE_KEY         Contents of the AuthKey_*.p8 private key

Optional environment:
  MACOS_SIGNING_IDENTITY        Developer ID Application identity override
  MACOS_KEYCHAIN_PASSWORD       Temporary CI keychain password
EOF
}

check_env() {
  local missing=()
  for name in "${required_env[@]}"; do
    if [[ -z "${!name:-}" ]]; then
      missing+=("${name}")
    fi
  done

  if (( ${#missing[@]} > 0 )); then
    printf 'Missing macOS notarization environment variables:\n' >&2
    printf '  %s\n' "${missing[@]}" >&2
    return 1
  fi
}

decode_certificate() {
  if printf '%s' "${MACOS_CERTIFICATE_P12_BASE64}" | base64 --decode > "${CERT_PATH}" 2>/dev/null; then
    return 0
  fi
  printf '%s' "${MACOS_CERTIFICATE_P12_BASE64}" | base64 -D > "${CERT_PATH}"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--check-env" ]]; then
  check_env
  exit 0
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "macOS notarization must run on a macOS runner." >&2
  exit 2
fi

check_env

for tool in base64 codesign ditto file security spctl xcrun; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "Required tool not found: ${tool}" >&2
    exit 2
  fi
done

if [[ ! -d "${APP_DIR}" ]]; then
  echo "App directory does not exist: ${APP_DIR}" >&2
  exit 2
fi

cleanup() {
  rm -f "${CERT_PATH}" "${API_KEY_PATH}"
  security delete-keychain "${KEYCHAIN_PATH}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

decode_certificate
printf '%s' "${APPLE_API_PRIVATE_KEY}" > "${API_KEY_PATH}"
chmod 600 "${API_KEY_PATH}"

security create-keychain -p "${KEYCHAIN_PASSWORD}" "${KEYCHAIN_PATH}"
security set-keychain-settings -lut 21600 "${KEYCHAIN_PATH}"
security unlock-keychain -p "${KEYCHAIN_PASSWORD}" "${KEYCHAIN_PATH}"
security import "${CERT_PATH}" -P "${MACOS_CERTIFICATE_PASSWORD}" -A -t cert -f pkcs12 -k "${KEYCHAIN_PATH}"
security set-key-partition-list -S apple-tool:,apple: -s -k "${KEYCHAIN_PASSWORD}" "${KEYCHAIN_PATH}"

SIGNING_IDENTITY="${MACOS_SIGNING_IDENTITY:-}"
if [[ -z "${SIGNING_IDENTITY}" ]]; then
  SIGNING_IDENTITY="$(
    security find-identity -v -p codesigning "${KEYCHAIN_PATH}" \
      | sed -n 's/.*"\(Developer ID Application:.*\)"/\1/p' \
      | head -n 1
  )"
fi

if [[ -z "${SIGNING_IDENTITY}" ]]; then
  echo "No Developer ID Application signing identity found." >&2
  exit 1
fi

echo "Signing Mach-O files in ${APP_DIR}"
while IFS= read -r -d '' path; do
  if file "${path}" | grep -q "Mach-O"; then
    codesign --force --options runtime --timestamp --sign "${SIGNING_IDENTITY}" "${path}"
  fi
done < <(find "${APP_DIR}" -type f -print0)

MAIN_EXECUTABLE="${APP_DIR}/OpenCut-Server"
if [[ -f "${MAIN_EXECUTABLE}" ]]; then
  codesign --verify --strict --verbose=2 "${MAIN_EXECUTABLE}"
  spctl --assess --type execute --verbose=2 "${MAIN_EXECUTABLE}" || true
fi

mkdir -p "$(dirname "${ZIP_PATH}")"
rm -f "${ZIP_PATH}"
ditto -c -k --keepParent "${APP_DIR}" "${ZIP_PATH}"

xcrun notarytool submit "${ZIP_PATH}" \
  --key "${API_KEY_PATH}" \
  --key-id "${APPLE_API_KEY_ID}" \
  --issuer "${APPLE_API_ISSUER_ID}" \
  --wait

if [[ "${APP_DIR}" == *.app ]]; then
  xcrun stapler staple "${APP_DIR}"
  xcrun stapler validate "${APP_DIR}"
else
  echo "Notarized ${ZIP_PATH}. ZIP archives cannot be stapled directly; Gatekeeper fetches the ticket online."
fi
