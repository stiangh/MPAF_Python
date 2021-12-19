in_name = 'data/mat/abu-urban-5.mat';
out_name = 'data/csv/abu-urban-5.csv';

load(in_name);

A = data(:,:,:);
A_map = map(:,:);

[x_dim, y_dim, s_dim] = size(A);

B = zeros(s_dim, x_dim*y_dim);
B_map = zeros(1, x_dim*y_dim);

for s = 1:s_dim
    for x = 1:x_dim
        for y = 1:y_dim
            B(s, (x-1)*y_dim+y) = A(x, y, s);
        end
    end
end

for x = 1:x_dim
    for y = 1:y_dim
        B_map((x-1)*y_dim+y) = A_map(x, y);
    end
end

dlmwrite(out_name, [x_dim, y_dim, s_dim]); % first line of csv will be dimensions
dlmwrite(out_name, B, '-append'); % next s_dim lines will be flattened x,y pixel array
dlmwrite(out_name, B_map, '-append'); % last line is x,y truth map
