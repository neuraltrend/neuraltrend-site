// Backtest date-range validation with keyboard-friendly MM/DD/YYYY entry.
document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById("backtest-form");
    const tickerSelect = document.getElementById("ticker");
    const startInput = document.getElementById("start");
    const endInput = document.getElementById("end");
    const startIsoInput = document.getElementById("start-iso");
    const endIsoInput = document.getElementById("end-iso");
    const startHint = document.getElementById("backtest-start-date-hint");
    const endHint = document.getElementById("backtest-end-date-hint");

    if (!form || !tickerSelect || !startInput || !endInput || !startIsoInput || !endIsoInput) return;

    let coverageStart = null;
    let coverageEnd = null;
    let boundsRequestId = 0;

    function shiftIsoDate(isoDate, days) {
        if (!isoDate) return "";
        const date = new Date(`${isoDate}T00:00:00Z`);
        if (Number.isNaN(date.getTime())) return "";
        date.setUTCDate(date.getUTCDate() + days);
        return date.toISOString().slice(0, 10);
    }

    function isoToDisplay(isoDate) {
        const match = String(isoDate || "").match(/^(\d{4})-(\d{2})-(\d{2})$/);
        return match ? `${match[2]}/${match[3]}/${match[1]}` : "";
    }

    function parseDisplayDate(rawValue) {
        const value = String(rawValue || "").trim();
        let match = value.match(/^(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{4})$/);
        if (!match) {
            const digits = value.replace(/\D/g, "");
            if (digits.length === 8) {
                match = [digits, digits.slice(0, 2), digits.slice(2, 4), digits.slice(4, 8)];
            }
        }
        if (!match) return null;

        const month = Number(match[1]);
        const day = Number(match[2]);
        const year = Number(match[3]);
        if (year < 1000 || month < 1 || month > 12 || day < 1 || day > 31) return null;

        const date = new Date(Date.UTC(year, month - 1, day));
        if (
            date.getUTCFullYear() !== year ||
            date.getUTCMonth() !== month - 1 ||
            date.getUTCDate() !== day
        ) {
            return null;
        }

        return `${String(year).padStart(4, "0")}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
    }

    function isPotentialPartialDate(rawValue) {
        const value = String(rawValue || "").trim();
        if (!value) return true;
        return /^[0-9/\-.\s]{1,10}$/.test(value);
    }

    function currentRange() {
        const startIso = parseDisplayDate(startInput.value);
        const endIso = parseDisplayDate(endInput.value);
        const startMax = endIso
            ? shiftIsoDate(endIso, -1)
            : coverageEnd ? shiftIsoDate(coverageEnd, -1) : "";
        const endMin = startIso
            ? shiftIsoDate(startIso, 1)
            : coverageStart ? shiftIsoDate(coverageStart, 1) : "";
        return {startIso, endIso, startMax, endMin};
    }

    function updateHints() {
        if (!coverageStart || !coverageEnd) return;
        const {startMax, endMin} = currentRange();
        if (startHint) {
            startHint.textContent = `Available: ${isoToDisplay(coverageStart)} – ${isoToDisplay(startMax)}`;
        }
        if (endHint) {
            endHint.textContent = `Available: ${isoToDisplay(endMin)} – ${isoToDisplay(coverageEnd)}`;
        }
    }

    function validateDateInputs({reportPartial = false} = {}) {
        startInput.setCustomValidity("");
        endInput.setCustomValidity("");
        startIsoInput.value = "";
        endIsoInput.value = "";

        if (!coverageStart || !coverageEnd) {
            startInput.setCustomValidity("Available backtest dates are still loading.");
            return false;
        }

        const startRaw = startInput.value.trim();
        const endRaw = endInput.value.trim();
        const {startIso, endIso, startMax, endMin} = currentRange();

        if (startRaw && !startIso) {
            if (reportPartial || !isPotentialPartialDate(startRaw)) {
                startInput.setCustomValidity("Enter a valid date as MM/DD/YYYY.");
            }
            return false;
        }
        if (endRaw && !endIso) {
            if (reportPartial || !isPotentialPartialDate(endRaw)) {
                endInput.setCustomValidity("Enter a valid date as MM/DD/YYYY.");
            }
            return false;
        }

        if (!startIso || !endIso) return false;

        if (startIso < coverageStart || startIso > startMax) {
            startInput.setCustomValidity(
                `Start date must be between ${isoToDisplay(coverageStart)} and ${isoToDisplay(startMax)}.`
            );
            return false;
        }

        if (endIso < endMin || endIso > coverageEnd) {
            endInput.setCustomValidity(
                `End date must be between ${isoToDisplay(endMin)} and ${isoToDisplay(coverageEnd)}.`
            );
            return false;
        }

        if (startIso >= endIso) {
            endInput.setCustomValidity("End date must be after the start date.");
            return false;
        }

        startIsoInput.value = startIso;
        endIsoInput.value = endIso;
        return true;
    }

    function onDateInput(input) {
        // Never clear or rewrite a partially typed year. Once a complete valid
        // date exists, normalize its display only on blur (not while typing).
        input.setCustomValidity("");
        validateDateInputs({reportPartial: false});
        updateHints();
    }

    [startInput, endInput].forEach(input => {
        input.addEventListener("input", () => onDateInput(input));
        input.addEventListener("blur", () => {
            const iso = parseDisplayDate(input.value);
            if (iso) input.value = isoToDisplay(iso);
            validateDateInputs({reportPartial: true});
            updateHints();
        });
    });

    async function loadBacktestDateBounds() {
        const requestId = ++boundsRequestId;
        const ticker = tickerSelect.value;

        coverageStart = null;
        coverageEnd = null;
        startIsoInput.value = "";
        endIsoInput.value = "";
        if (startHint) startHint.textContent = "Loading available dates…";
        if (endHint) endHint.textContent = "Loading available dates…";

        try {
            const response = await fetch(
                `/backtest/date-range?ticker=${encodeURIComponent(ticker)}`,
                {cache: "no-store"}
            );
            const payload = await response.json();

            if (requestId !== boundsRequestId) return;
            if (!response.ok) throw new Error(payload.error || "Backtest date range is unavailable.");

            coverageStart = payload.coverage_start;
            coverageEnd = payload.coverage_end;

            // Preserve typed dates only when they remain valid for this asset.
            const startIso = parseDisplayDate(startInput.value);
            const endIso = parseDisplayDate(endInput.value);
            if (startIso && (startIso < coverageStart || startIso > shiftIsoDate(coverageEnd, -1))) {
                startInput.value = "";
            }
            if (endIso && (endIso < shiftIsoDate(coverageStart, 1) || endIso > coverageEnd)) {
                endInput.value = "";
            }

            validateDateInputs({reportPartial: false});
            updateHints();
        } catch (error) {
            console.error("Could not load backtest date range:", error);
            coverageStart = null;
            coverageEnd = null;
            if (startHint) startHint.textContent = "Available dates could not be loaded.";
            if (endHint) endHint.textContent = "Available dates could not be loaded.";
            startInput.setCustomValidity("Available backtest dates could not be loaded for this asset.");
            endInput.setCustomValidity("Available backtest dates could not be loaded for this asset.");
        }
    }

    tickerSelect.addEventListener("change", loadBacktestDateBounds);

    window.getBacktestIsoDates = function() {
        validateDateInputs({reportPartial: true});
        return {
            start: startIsoInput.value,
            end: endIsoInput.value
        };
    };

    window.validateBacktestDateRange = function() {
        return Boolean(validateDateInputs({reportPartial: true}));
    };

    form.addEventListener("submit", function(event) {
        if (!window.validateBacktestDateRange()) {
            event.preventDefault();
            form.reportValidity();
        }
    });

    loadBacktestDateBounds();
});
