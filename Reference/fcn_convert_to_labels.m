% this function is to convert the output from matrix to labels

function output_labels = fcn_convert_to_labels(output_matrix)

output_labels = [];

for i = 1:size(output_matrix,2)

    max_value = max(output_matrix(:,i));

    for j = 1:size(output_matrix,1)

        if output_matrix(j,i) == max_value

            output_labels = [output_labels j];

            % 2017/12/01: fix a bug when same confidence values occurs by adding an addition "break"

            break;

        end

    end

end

end

