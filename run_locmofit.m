% Uncomment the two lines below for a fresh start
% terminate(pyenv)
% clear

% Start Python
pyenv(Version="/Users/lancexwq/Ries Lab/LocMoFitPy2/.pixi/envs/default/bin/python", ExecutionMode= "OutOfProcess");

% Load data from CSV
data = readmatrix("./notebooks/data/simulated_data_spcap.csv");
locs = data(:, 1:3);
loc_precisions = data(:,4:6);

tic
% Import locmofitpy
api = py.importlib.import_module("locmofitpy");

init_params = py.dict(pyargs("c", 0.02));
% freeze = py.tuple("c");
freeze = py.tuple();

% Run the code to fit
res = api.run_locmofit( ...
    "SphericalCap", ...
    locs, ...
    loc_precisions, ...
    pyargs( ...
        "init_params", init_params, ...
        "freeze", freeze, ...
        "max_iter", int64(500), ...
        "spacing", 3, ...
        "dtype", "float32", ...
        "tol", 1e-6 ...
    ) ...
);

% Collect the result
losses = double(res{"losses"});
positions  = double(res{"model_points"});
parameters = res{"parameters"};
toc

% Simple visualization
ground_truth = readmatrix("./notebooks/data/ground_truth_spcap.csv");
scatter3(positions(:,1), positions(:,2), positions(:,3))
hold on
scatter3(ground_truth(:,1), ground_truth(:,2), ground_truth(:,3))
daspect([1 1 1])
legend('model points','localizations')
hold off