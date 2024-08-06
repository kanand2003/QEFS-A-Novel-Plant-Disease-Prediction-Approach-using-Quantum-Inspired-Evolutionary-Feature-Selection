function [Best_pos, Best_score, conv] = ACO_WOA(SearchAgents_no, Max_iteration, lb, ub, dim, fobj, X, Y)

    rng(1);
    conv = [];
    iteration = 1;

    % Initialize ant positions and velocities
    ants_pos = round(rand(SearchAgents_no, dim));
    ants_vel = rand(SearchAgents_no, dim);

    % Evaluate fitness
    fitness = zeros(SearchAgents_no, 1);
    for i = 1:SearchAgents_no
        fitness(i) = fobj(ants_pos(i, :), X, Y);
    end

    % Find the initial best solution
    [Best_score, tindex] = max(fitness);
    Best_pos = ants_pos(tindex, :);
    conv = [conv Best_score];

    % Set WOA parameters
    woa_max_iteration = Max_iteration;  % Set maximum iteration for WOA

    while iteration <= Max_iteration
        iteration = iteration + 1;

        % ACO Exploration Phase
        ants_pos = aco_explore(ants_pos, fitness, lb, ub, dim);
        % Evaluate fitness of new ant positions
        fitness = zeros(SearchAgents_no, 1);
        for i = 1:SearchAgents_no
            fitness(i) = fobj(ants_pos(i, :), X, Y);
        end

        % WOA Velocity Update Phase
        ants_vel = woa_update_velocity(ants_vel, Best_pos, woa_max_iteration, iteration);

        % Update ant positions based on velocity
        ants_pos = ants_pos + ants_vel;

        % Ensure that the positions are within the search space bounds
        ants_pos = max(min(ants_pos, ub), lb);

        % Update best solution
        [maxFitness, maxIndex] = max(fitness);
        if maxFitness > Best_score
            Best_score = maxFitness;
            Best_pos = ants_pos(maxIndex, :);
        end

        % Store the best score for plotting convergence
        conv(iteration) = Best_score;

        % Display the iteration and best score
        disp(['Iteration ' num2str(iteration - 1) ', Best Score: ' num2str(Best_score)]);
    end

end

function new_positions = aco_explore(ants_pos, fitness, lb, ub, dim)
    [num_ants, ~] = size(ants_pos);

    % Initialize matrix to store pheromones on features
    pheromones = ones(num_ants, dim);

    % Iterate over each ant to construct solutions probabilistically
    for ant = 1:num_ants
        current_feature = randi(dim); % Start from a random feature
        selected_features = zeros(1, dim); % Initialize selected features
        
        for step = 1:dim
            % Calculate cumulative probabilities
            probabilities = pheromones(ant, :);
            probabilities(current_feature) = 0;  % Prevent choosing the current feature
            probabilities = probabilities / sum(probabilities);

            % Choose the next feature based on probabilities
            selected_feature_index = randsample(1:dim, 1, true, probabilities);

            % Update selected features
            selected_features(selected_feature_index) = 1;

            % Move to the next feature
            current_feature = selected_feature_index;
        end

        % Update ant's position with the selected features
        ants_pos(ant, :) = selected_features;
    end

    % Ensure that the positions are within the search space bounds
    new_positions = max(min(ants_pos, ub), lb);
end

function new_velocities = woa_update_velocity(ants_vel, Best_pos, woa_max_iteration, current_iteration)
    a = 2 - current_iteration * (2 / woa_max_iteration); % a decreases linearly from 2 to 0
    r1 = rand(); % random number between 0 and 1
    r2 = rand(); % random number between 0 and 1

    A = 2 * a * r1 - a;
    C = 2 * r2;

    b = 1;
    l = (a - 1) * rand() + 1;

    p = rand(); % random number between 0 and 1

    new_velocities = zeros(size(ants_vel));

    for ant = 1:size(ants_vel, 1)
        if p < 0.5
            if abs(A) >= 1
                % Update velocity using WOA equation
                new_velocities(ant, :) = rand() * A * C + rand() * Best_pos - ants_vel(ant, :);
            else
                % Update velocity using WOA equation
                new_velocities(ant, :) = abs(Best_pos - ants_vel(ant, :)) * exp(b * l) .* cos(2 * pi * l) + Best_pos - ants_vel(ant, :);
            end
        else
            % Update velocity using WOA equation
            new_velocities(ant, :) = abs(Best_pos - ants_vel(ant, :)) * exp(b * l) * cos(2 * pi * l) + Best_pos - ants_vel(ant, :);
        end
    end
end
