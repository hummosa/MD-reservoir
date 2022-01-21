N_TRIALS = 350;
subjs = ["01", '02', '04', '06', '09', '10', '12', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '28', '29', '30', '31', '32', '33', '35', '36', "37"];

Y = zeros(numel(subjs), N_TRIALS);
for idx=1:numel(subjs)
    subj = subjs(idx);
    load(sprintf('data/human_results/Sub%s/Result.mat', subj))
    
    y = Result(:,9) - 1;
    y(y == -1) = nan;
    Y(idx,:) = abs(y - 1);
end

contextchange = find(ischange(Result(:,2)));

X = mean(Y,1,'omitnan');
transition9030_1 = X(contextchange(1)-10:contextchange(1)+30);
transition9030_2 = X(contextchange(7)-10:contextchange(7)+30);
X9030 = mean([transition9030_1;transition9030_2]);

transition7010_1 = X(contextchange(4)-10:contextchange(4)+30);
transition7010_2 = X(contextchange(8)-10:contextchange(8)+30);
X7010 = mean([transition7010_1;transition7010_2]);

X9030_7010 = mean([X9030;X7010]);

figure;
subplot(2,2,1);
hold on;
plot(-10:30,X9030);
vline(0,'k--');
xlabel('Trials from block change');
ylabel('Performance');
title('90-30 changes');
subplot(2,2,2);
hold on;
plot(-10:30,X7010);
vline(0,'k--');
title('70-30 changes');
subplot(2,2,3);
plot(-10:30,X9030_7010);
vline(0,'k--');
title('90-30 and 70-10 changes')
subplot(2,2,4);
boxchart([X9030(11:end)' X7010(11:end)' X9030_7010(11:end)'],'Notch','on');
xticklabels({'9030', '7010', '9030 and 7010'});
ylabel('Ratio correct after switch');

function plot_schedule()
load(sprintf('data/human_results/Sub%s/Result.mat', "01"))
contextchange = find(ischange(Result(:,2)));
text(10, 0.2, sprintf("%s", num2str(Result(1,2))))
for idx=1:numel(contextchange)
    trialidx = contextchange(idx);
    vline(trialidx)
    text(trialidx + 10, 0.2, sprintf("%s", num2str(Result(trialidx,2))))
end
end
