% Uncomment the two lines below for a fresh start
% terminate(pyenv)
% clear

% Start Python
pyenv(Version=fullfile(pwd, ".pixi/envs/default/bin/python"), ExecutionMode= "OutOfProcess");

% Load data from CSV
data = readmatrix("./notebooks/data/simulated_data_spcap.csv");
locs = data(:, 1:3);
loc_precisions = data(:,4:6);

tic
% Import locmofitpy
api = py.importlib.import_module("locmofitpy");

% Starting values and priors, one ParamToFit per quantity (the docstring of run_locmofit lists them); a quantity not named starts from the data, with its default prior
params = py.dict(pyargs("kappa", api.ParamToFit(0.4)));
% params = py.dict(pyargs("kappa", api.ParamToFit(0.4, api.Fixed()), "axis", api.ParamToFit(py.tuple({0, 0}))));

% Run the code to fit
res = api.run_locmofit( ...
    "SphericalCap", ...
    locs, ...
    loc_precisions, ...
    pyargs( ...
        "params", params, ...
        "max_iter", int64(500), ...
        "spacing", 3, ...
        "dtype", "float32", ...
        "tol", 1e-6 ...
    ) ...
);

% Collect the result
loss = double(res{"loss"});
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