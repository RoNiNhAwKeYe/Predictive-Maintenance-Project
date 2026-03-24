// TIME
document.getElementById("time").innerText = new Date().toLocaleString();

// RANDOM STATUS SIMULATION
function updateStatus() {
    const states = ["Normal", "Bearing Fault", "Gear Fault"];
    const state = states[Math.floor(Math.random() * states.length)];

    document.getElementById("prediction").innerText = state;

    if (state === "Normal") {
        document.getElementById("status").innerText = "Normal";
        document.getElementById("alert").innerText = "None";
        document.getElementById("alerts").innerText = "No issues detected";
    } else {
        document.getElementById("status").innerText = "Faulty";
        document.getElementById("alert").innerText = "Critical";
        document.getElementById("alerts").innerText =
            state + " detected – inspect immediately";
    }
}

updateStatus();
function updateStatus() {
    const states = ["Normal", "Bearing Fault", "Gear Fault"];
    const state = states[Math.floor(Math.random() * states.length)];

    const statusEl = document.getElementById("status");
    const alertEl = document.getElementById("alert");

    document.getElementById("prediction").innerText = state;

    if (state === "Normal") {
        statusEl.innerText = "Normal";
        statusEl.style.color = "#22c55e";

        alertEl.innerText = "None";
        alertEl.style.color = "#22c55e";

        document.getElementById("alerts").innerText = "No issues detected";
    } else {
        statusEl.innerText = "Faulty";
        statusEl.style.color = "#ef4444";

        alertEl.innerText = "Critical";
        alertEl.style.color = "#ef4444";

        document.getElementById("alerts").innerText =
            state + " detected – inspect immediately";
    }
}