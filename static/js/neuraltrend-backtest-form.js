// Backtest native date-range controls with keyboard-safe manual entry.
document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById("backtest-form");
    const tickerSelect = document.getElementById("ticker");
    const startInput = document.getElementById("start");
    const endInput = document.getElementById("end");

    if (!form || !tickerSelect || !startInput || !endInput) return;

    let coverageStart = null;
    let coverageEnd = null;
    let committedStart = "";
    let committedEnd = "";
    let boundsRequestId = 0;

    const keyboardEditing = new WeakSet();

    function shiftIsoDate(isoDate, days) {
        if (!isoDate) return "";
        const date = new Date(`${isoDate}T00:00:00Z`);
        if (Number.isNaN(date.getTime())) return "";
        date.setUTCDate(date.getUTCDate() + days);
        return date.toISOString().slice(0, 10);
    }

    function isIsoDate(value) {
        return /^\d{4}-\d{2}-\d{2}$/.test(String(value || ""));
    }

    function absoluteStartMax() {
        return coverageEnd ? shiftIsoDate(coverageEnd, -1) : "";
    }

    function absoluteEndMin() {
        return coverageStart ? shiftIsoDate(coverageStart, 1) : "";
    }

    function startMaxForCommittedRange() {
        return committedEnd
            ? shiftIsoDate(committedEnd, -1)
            : absoluteStartMax();
    }

    function endMinForCommittedRange() {
        return committedStart
            ? shiftIsoDate(committedStart, 1)
            : absoluteEndMin();
    }

    function applyConstraintsToInput(input, minValue, maxValue) {
        if (keyboardEditing.has(input)) return;

        if (minValue) input.min = minValue;
        else input.removeAttribute("min");

        if (maxValue) input.max = maxValue;
        else input.removeAttribute("max");
    }

    function restoreDateConstraints() {
        if (!coverageStart || !coverageEnd) return;

        applyConstraintsToInput(
            startInput,
            coverageStart,
            startMaxForCommittedRange()
        );
        applyConstraintsToInput(
            endInput,
            endMinForCommittedRange(),
            coverageEnd
        );
    }

    function finishKeyboardEditing(input = null) {
        if (input) keyboardEditing.delete(input);
        else {
            keyboardEditing.delete(startInput);
            keyboardEditing.delete(endInput);
        }
        restoreDateConstraints();
    }

    function beginKeyboardEditing(input) {
        keyboardEditing.add(input);

        // Native Chromium date controls can validate the partially typed year
        // against min/max (for example "20" while entering "2024") and reset
        // the field. Suspend only this field's bounds while numeric segments
        // are being typed. The picker still receives the bounds because they
        // are restored on pointerdown before it opens.
        input.removeAttribute("min");
        input.removeAttribute("max");
        input.setCustomValidity("");
    }

    function valueInsideAbsoluteCoverage(value, kind) {
        if (!value || !coverageStart || !coverageEnd) return false;

        if (kind === "start") {
            return value >= coverageStart && value <= absoluteStartMax();
        }
        return value >= absoluteEndMin() && value <= coverageEnd;
    }

    function commitStartValue() {
        const value = startInput.value;
        committedStart = valueInsideAbsoluteCoverage(value, "start") ? value : "";

        if (committedStart && committedEnd && committedEnd <= committedStart) {
            committedEnd = "";
            endInput.value = "";
        }
    }

    function commitEndValue() {
        const value = endInput.value;
        committedEnd = valueInsideAbsoluteCoverage(value, "end") ? value : "";

        if (committedStart && committedEnd && committedStart >= committedEnd) {
            committedStart = "";
            startInput.value = "";
        }
    }

    function validateDateRange({report = false} = {}) {
        finishKeyboardEditing();
        startInput.setCustomValidity("");
        endInput.setCustomValidity("");

        if (!coverageStart || !coverageEnd) {
            startInput.setCustomValidity("Available backtest dates are still loading.");
            if (report) startInput.reportValidity();
            return false;
        }

        const startValue = startInput.value;
        const endValue = endInput.value;

        if (!startValue) {
            startInput.setCustomValidity("Choose a start date.");
            if (report) startInput.reportValidity();
            return false;
        }

        if (!endValue) {
            endInput.setCustomValidity("Choose an end date.");
            if (report) endInput.reportValidity();
            return false;
        }

        if (!isIsoDate(startValue) || !valueInsideAbsoluteCoverage(startValue, "start")) {
            startInput.setCustomValidity("Choose a start date within the available range.");
            if (report) startInput.reportValidity();
            return false;
        }

        if (!isIsoDate(endValue) || !valueInsideAbsoluteCoverage(endValue, "end")) {
            endInput.setCustomValidity("Choose an end date within the available range.");
            if (report) endInput.reportValidity();
            return false;
        }

        if (startValue >= endValue) {
            endInput.setCustomValidity("End date must be after the start date.");
            if (report) endInput.reportValidity();
            return false;
        }

        committedStart = startValue;
        committedEnd = endValue;
        restoreDateConstraints();
        return true;
    }

    function handleCommittedChange(input) {
        finishKeyboardEditing(input);
        input.setCustomValidity("");

        if (input === startInput) commitStartValue();
        else commitEndValue();

        restoreDateConstraints();
    }

    const pickerPointerActive = new WeakSet();

    function pointerIsOnNativePicker(input, event) {
        if (
            !event ||
            typeof event.clientX !== "number" ||
            typeof input.getBoundingClientRect !== "function"
        ) {
            return false;
        }

        const rect = input.getBoundingClientRect();

        // Chromium/WebKit place the native calendar affordance at the far
        // right of a date input. Keep this zone deliberately narrow so
        // clicking the YYYY segment still enters keyboard-edit mode.
        return event.clientX >= (rect.right - 32);
    }

    [startInput, endInput].forEach(input => {
        /*
          IMPORTANT:
          Native Chromium date inputs can reject a partially typed year before
          a normal keydown handler has a chance to help when min/max are active.
          Example: while entering 2024, the intermediate "20" can be considered
          outside a min year such as 2000 and the control resets.

          Therefore bounds are suspended as soon as the user focuses/clicks the
          editable date segments, not merely after the first digit is typed.
          They are restored before the native calendar picker opens, so dates
          outside the allowed range remain greyed out and unselectable.
        */
        input.addEventListener("pointerdown", event => {
            if (pointerIsOnNativePicker(input, event)) {
                pickerPointerActive.add(input);
                finishKeyboardEditing(input);
            } else {
                pickerPointerActive.delete(input);
                beginKeyboardEditing(input);
            }
        });

        input.addEventListener("focus", () => {
            if (!pickerPointerActive.has(input)) {
                beginKeyboardEditing(input);
            }
        });

        input.addEventListener("click", event => {
            if (pointerIsOnNativePicker(input, event)) {
                finishKeyboardEditing(input);
            } else {
                beginKeyboardEditing(input);
            }

            pickerPointerActive.delete(input);
        });

        input.addEventListener("keydown", event => {
            // Keyboard focus (Tab into the field) also needs bounds suspended
            // before numeric segment editing. Alt+Down is treated as a picker
            // request, so restore constraints for that interaction.
            if (event.altKey && event.key === "ArrowDown") {
                finishKeyboardEditing(input);
                return;
            }

            if (
                /^\d$/.test(event.key) ||
                event.key === "Backspace" ||
                event.key === "Delete"
            ) {
                beginKeyboardEditing(input);
            }
        }, true);

        input.addEventListener("change", () => {
            /*
              Chrome may emit change events while individual MM/DD/YYYY
              segments are still being edited. Do NOT restore min/max here
              during keyboard entry; doing so is what causes the next year
              digit (notably the 0 in 2024) to reset the whole field.

              A calendar selection is not in keyboard-edit mode, so it still
              commits immediately through this same event.
            */
            if (keyboardEditing.has(input)) return;
            handleCommittedChange(input);
        });

        input.addEventListener("blur", () => {
            pickerPointerActive.delete(input);
            handleCommittedChange(input);
        });
    });

    async function loadBacktestDateBounds() {
        const requestId = ++boundsRequestId;
        const ticker = tickerSelect.value;

        coverageStart = null;
        coverageEnd = null;
        committedStart = "";
        committedEnd = "";
        finishKeyboardEditing();

        startInput.disabled = true;
        endInput.disabled = true;
        startInput.setCustomValidity("");
        endInput.setCustomValidity("");

        try {
            const response = await fetch(
                `/backtest/date-range?ticker=${encodeURIComponent(ticker)}`,
                {cache: "no-store"}
            );
            const payload = await response.json();

            if (requestId !== boundsRequestId) return;
            if (!response.ok) {
                throw new Error(payload.error || "Backtest date range is unavailable.");
            }

            coverageStart = payload.coverage_start;
            coverageEnd = payload.coverage_end;

            if (!coverageStart || !coverageEnd || coverageStart >= coverageEnd) {
                throw new Error("Backtest date coverage is invalid for this asset.");
            }

            // Preserve selections only when they remain valid for the newly
            // chosen ticker. Otherwise leave the native date field blank.
            if (valueInsideAbsoluteCoverage(startInput.value, "start")) {
                committedStart = startInput.value;
            } else {
                startInput.value = "";
            }

            if (valueInsideAbsoluteCoverage(endInput.value, "end")) {
                committedEnd = endInput.value;
            } else {
                endInput.value = "";
            }

            if (committedStart && committedEnd && committedStart >= committedEnd) {
                endInput.value = "";
                committedEnd = "";
            }

            restoreDateConstraints();
            startInput.disabled = false;
            endInput.disabled = false;
        } catch (error) {
            console.error("Could not load backtest date range:", error);
            coverageStart = null;
            coverageEnd = null;
            committedStart = "";
            committedEnd = "";
            startInput.removeAttribute("min");
            startInput.removeAttribute("max");
            endInput.removeAttribute("min");
            endInput.removeAttribute("max");
            startInput.disabled = false;
            endInput.disabled = false;
            startInput.setCustomValidity(
                "Available backtest dates could not be loaded for this asset."
            );
            endInput.setCustomValidity(
                "Available backtest dates could not be loaded for this asset."
            );
        }
    }

    tickerSelect.addEventListener("change", loadBacktestDateBounds);

    window.getBacktestIsoDates = function() {
        return {
            start: startInput.value || "",
            end: endInput.value || ""
        };
    };

    window.validateBacktestDateRange = function() {
        return Boolean(validateDateRange({report: false}));
    };

    form.addEventListener("submit", function(event) {
        if (!validateDateRange({report: false})) {
            event.preventDefault();
            form.reportValidity();
        }
    });

    loadBacktestDateBounds();
});
