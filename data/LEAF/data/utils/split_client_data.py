import argparse
import json
import os
import os.path as osp

from constants import DATASETS


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name',
                        help='name of dataset to parse; default: sent140;',
                        type=str,
                        choices=DATASETS,
                        default='sent140')

    args = parser.parse_args()

    parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_fd = os.path.join(parent_path, args.name, 'data')

    input_fds = {
        'train': osp.join(data_fd, 'train'),
        'test': osp.join(data_fd, 'test')
    }
    output_fd = os.path.join(data_fd, 'clients_split')
    os.makedirs(output_fd)

    output_fd_data = osp.join(output_fd, 'data')
    output_fd_clients = osp.join(output_fd, 'clients')
    os.mkdir(output_fd_data)
    os.mkdir(output_fd_clients)

    fig_id = 0
    clients = dict()
    for key, input_fd in input_fds.items():
        file_names = os.listdir(input_fd)
        input_fps = [osp.join(input_fd, file_name) for file_name in file_names]

        for input_fp in input_fps:
            print('processing {}'.format(input_fp))
            input_data = json.load(open(input_fp, 'r'))
            input_data = input_data['user_data']

            for client_name, client_data in input_data.items():
                client_data_part = []
                for x, y in zip(client_data['x'], client_data['y']):
                    fig_name = '{}.json'.format(fig_id)
                    assert not osp.exists(osp.join(output_fd_data, fig_name)), '{} already exists in {}'.format(
                        fig_name, output_fd_data)
                    json.dump(x, open(osp.join(output_fd_data, fig_name), 'w'))
                    client_data_part.append((fig_name, y))

                    fig_id += 1
                client_data_entry = clients.get(client_name, dict())
                client_data_entry[key] = client_data_part
                clients[client_name] = client_data_entry

    for client_name, client_data in clients.items():
        json.dump(client_data, open(osp.join(output_fd_clients, '{}.json'.format(client_name)), 'w'))


if __name__ == '__main__':
    main()
